_base_ = ['../PixArt-alpha/configs/PixArt_xl2_internal.py']

# Dataset. start_training_ffhq.sh passes --data_root (default dataset/ID-FFHQ).
data_root = 'dataset/ID-FFHQ'
image_list_json = ['data_info_FFHQ.json']
zero_text_training = False  # if True, skip T5 prompt features

data = dict(
    type='InternalDataControl',
    root='InternData',
    image_list_json=image_list_json,
    # JSON lives at dataset root (same level as images), not InternData/partition_filter/
    metadata_json_subdir='dataset_root',
    transform='default_train',
    load_vae_feat=True,
    # long_prompt_ratio in [0, 1]: per-sample probability of the long T5 branch
    # (prompt_feature_long); otherwise use prompt_feature (short prompt)
    long_prompt_ratio=0.3,
    prompt_feature_dir='prompt_feature',
    prompt_feature_long_dir='prompt_feature_long',
    zero_text_training=zero_text_training,
    controlnet_modality='Fusion',
    controlnet_feat_dir='edges_feature_1024',
    controlnet_feat_dir_2='lq_feature_1024',
    # LQ features from preprocess_dataset.py: InternData/lq_feature_1024/
)
use_frequency_control_fusion = True  # high-frequency edges + low-frequency LQ
freq_fusion_unfreeze_step = -1  # unfreeze r0/k; -1 = auto at 30% of total steps
adaface_start_epoch = 1  # skip AdaFace loss before this epoch
freq_fusion_param_warmup_steps = 2000  # gradient warmup steps for fusion params
freq_fusion_param_warmup_start_factor = 0.1  # start factor; linear ramp to 1.0
image_size = 1024

# Model
model = 'PixArtMS_XL_2'
fp32_attention = False  # set True if loss becomes NaN
load_from = 'models/PixArt-XL-2-1024-MS.pth'  # pretrained PixArt checkpoint
vae_pretrained = "models/sd-vae-ft-ema"
window_block_indexes = []
window_size = 0
use_rel_pos = False
lewei_scale = 2.0

# Training. Output dir is set by start_training_ffhq.sh --work-dir (default work_dirs/controlnet_ffhq).
num_workers = 8
train_batch_size = 16  # lower if GPU memory is tight (e.g. 4 on 24GB, 8 on 40GB)
num_epochs = 40
gradient_accumulation_steps = 4  # effective batch = train_batch_size * gradient_accumulation_steps * num_gpus
grad_checkpointing = True  # save memory
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule = 'constant'
lr_schedule_args = dict(num_warmup_steps=0)
save_model_epochs = 5
save_model_steps = 1000
log_interval = 20

# ControlNet
copy_blocks_num = 13
class_dropout_prob = 0.5
train_ratio = 0.81  # 81% train
val_ratio = 0.19  # 19% val; train_ratio + val_ratio should be 1.0
val_batch_size = 4  # can be smaller than train_batch_size to save memory
val_interval = 1  # validate every N epochs (1 = every epoch)
val_steps = 500  # also validate every N steps; None = epoch end only

# Other
mixed_precision = 'fp16'
scale_factor = 0.18215  # VAE scale
train_sampling_steps = 1000
model_max_length = 120  # T5 max tokens
aspect_ratio_type = None  # single-resolution
multi_scale = False

# AdaFace cosine-similarity loss
use_adaface_loss = True
adaface_model_path = 'models/adaface_ir101_webface12m.ckpt'
adaface_loss_weight = 0.1  # typically 0.05-0.2
adaface_loss_freq = 10  # compute every N steps (1 = every step)
adaface_max_timestep = 200  # only when timestep is below this (later denoising)
adaface_quality_gate = True  # skip samples where teacher detect/align fails
adaface_warmup_steps = 4000  # disable AdaFace during warmup
adaface_ramp_steps = 1500  # then linearly increase AdaFace weight
adaface_use_sigmoid_weight = True  # sigmoid weight around the same-person threshold
adaface_same_person_threshold = 0.38482  # T: AdaFace same-identity threshold
adaface_sigmoid_alpha = 10.0  # slope; larger = sharper transition
adaface_sigmoid_w_min = 0.0
adaface_sigmoid_w_max = 1.0
