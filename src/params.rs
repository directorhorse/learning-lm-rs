use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            let tensorname;
            if config.tie_word_embeddings {
                tensorname = match name {
                    "lm_head" => "lm_head.weight",
                    "embedding_table" => "lm_head.weight",
                    "rms_out_w" => "model.norm.weight",
                    _ => "",
                };
            } else {
                tensorname = match name {
                    "lm_head" => "lm_head.weight",
                    "embedding_table" => "model.embed_tokens.weight",
                    "rms_out_w" => "model.norm.weight",
                    _ => "",
                };
            }

            let tensorview = safetensor.tensor(if tensorname != "" {tensorname} else {name}).unwrap();
            let vec_u8 = tensorview.data().to_vec();
            let mut vec_f32 = Vec::<f32>::new();
            for i in 0..(vec_u8.len() / 4) {
                let mut v: [u8; 4] = [0, 0, 0, 0];
                for j in 0..4 {
                    v[j] = vec_u8[i*4 + j];
                }
                vec_f32.push(f32::from_le_bytes(v));
            }
            Tensor::new(vec_f32, &tensorview.shape().to_vec())
        };
        
        let get_tensors = |name: &str| {
            let n_layers = config.num_hidden_layers;
            let mut tensor_vec = Vec::<Tensor<f32>>::with_capacity(n_layers);

            let tensorname = match name {
                "rms_att_w" => "input_layernorm.weight",
                "wq" => "self_attn.q_proj.weight",
                "wk" => "self_attn.k_proj.weight",
                "wv" => "self_attn.v_proj.weight",
                "wo" => "self_attn.o_proj.weight",
                "rms_ffn_w" => "post_attention_layernorm.weight",
                "w_up" => "mlp.up_proj.weight",
                "w_gate" => "mlp.gate_proj.weight",
                "w_down" => "mlp.down_proj.weight",
                _ => "",
            };

            for n in 0..n_layers {
                tensor_vec.push(get_tensor(&format!("model.layers.{n}.{tensorname}")));
            }
            return tensor_vec;
        };

        LLamaParams {
            embedding_table: get_tensor("embedding_table"),
            rms_att_w: get_tensors("rms_att_w"),
            wq: get_tensors("wq"),
            wk: get_tensors("wk"),
            wv: get_tensors("wv"),
            wo: get_tensors("wo"),
            rms_ffn_w: get_tensors("rms_ffn_w"),
            w_up: get_tensors("w_up"),
            w_gate: get_tensors("w_gate"),
            w_down: get_tensors("w_down"),
            rms_out_w: get_tensor("rms_out_w"),
            lm_head: get_tensor("lm_head"),
        }
    }
}