use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, masked_softmax, matmul_transb, rms_norm, swiglu, transb};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // todo!("self_attention(...)");
            self_attention(&mut hidden_states, &mut att_scores, &q, &full_k, &full_v, self.n_kv_h, n_groups, seq_len, total_seq_len, 
                self.dqkv
            );
            matmul_transb(
                &mut residual, 
                1.0, 
                &hidden_states, 
                &self.params.wo[layer], 
                1.0
            );
            mlp(
                &mut residual, 
                &mut hidden_states, 
                &mut gate_buf, 
                &mut up_buf, 
                &self.params.w_up[layer], 
                &self.params.w_down[layer], 
                &self.params.w_gate[layer], 
                &self.params.rms_ffn_w[layer], 
                self.eps
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();
        
        // todo!("实现文本生成");
        let mut cache = self.new_cache();
        // initial step for cache
        let mut output = self.forward(&Tensor::<u32>::new(token_ids.to_vec(), &vec![token_ids.len()]), &mut cache);
        let mut output_id = OP::random_sample(&output, top_p, top_k, temperature);

        // output.print();

        result.push(output_id);
        // generate next token
        let mut next_input = Tensor::<u32>::new(vec![output_id], &vec![1]);
        for i in 0..max_len {
            output = self.forward(&next_input, &mut cache);
            output_id = OP::random_sample(&output, top_p, top_k, temperature);
            if output_id == self.eos_token_id {
                break;
            } else {
                result.push(output_id);
                next_input = Tensor::<u32>::new(vec![output_id], &vec![1]);
            }
        }
        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let scores = unsafe { att_scores.data_mut() };
    for head_index in 0..n_kv_h{//按照n_kv_h和n_groups来划分
        for group_index in 0..n_groups{//(seq,dqkv)x(total_seq,dqkv).T,得到score的坐标(i,j,...,...)
            for i in 0..seq_len{
                for j in 0..total_seq_len{
                    let mut temp = 0.0;
                    //此时q对应的索引是i*n_kv_h * n_groups * dqkv + head_index * n_groups * dqkv + group_index *dqkv + z
                    //k对应的索引是j*n_kv_h * dqkv + head_index * dqkv + z
                    //在score里的坐标是(head_index,group_index,i,j)，即head_index*n_groups*seq*total_seq + group_index*seq*total_seq+i*total_seq+j
                    for z in 0..dqkv{
                        let q_index = i*n_kv_h * n_groups * dqkv + head_index * n_groups * dqkv + group_index *dqkv + z;
                        let k_index = j*n_kv_h * dqkv + head_index * dqkv + z;
                        temp += q.data()[q_index] * k.data()[k_index];
                    }
                    let score_index = head_index*n_groups*seq_len*total_seq_len + group_index*seq_len*total_seq_len+i*total_seq_len+j;
                    scores[score_index] = temp/(dqkv as f32).sqrt();
                }
            }
        }
    }
    // attn = softmax(score)
    OP::masked_softmax(att_scores);
    // attn_V = attn @ V
    let hidden = unsafe { hidden_states.data_mut() };
    let v_data = v.data();
    for head_index in 0..n_kv_h{
        for group_index in 0..n_groups{
            for i in 0..seq_len{
                for j in 0..dqkv{
                    let mut temp = 0.0;
                    for z in 0..total_seq_len{
                        //(n_kv_h, n_groups, seq, total_seq)
                        //在attn_scores里的坐标为(head_index,group_index,i,z)，即head_index*n_groups*seq_len*total_seq_len+group_index*seq_len*total_seq_len+i*total_seq_len+z
                        //在v里的坐标为z*n_kv_h * dqkv + head_index*dqkv + j
                        let attn_index = head_index*n_groups*seq_len*total_seq_len+group_index*seq_len*total_seq_len+i*total_seq_len+z;
                        let v_index = z*n_kv_h * dqkv + head_index*dqkv + j;
                        temp += att_scores.data()[attn_index]*v_data[v_index];
                    }
                    //在hidden_states里的位置为i*n_kv_h * n_groups * dqkv + head_index * n_groups * dqkv + group_index * dqkv +j
                    let hidden_states_index = i*n_kv_h * n_groups * dqkv + head_index * n_groups * dqkv + group_index * dqkv +j;
                    hidden[hidden_states_index] = temp;
                } 
            }
        }
    }

}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {

    OP::rms_norm(hidden_states, residual, rms_w, eps);
    OP::matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);
    OP::matmul_transb(up, 0.0, hidden_states, w_up, 1.0);
    OP::swiglu(up, gate);
    OP::matmul_transb(hidden_states, 0.0, up, w_down, 1.0);
    OP::matmul_transb(residual, 1.0, up, w_down, 1.0);
    // todo!("Implement mlp");
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );
    for data in hidden_states.data(){
        print!("{}",data)
    }
    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
