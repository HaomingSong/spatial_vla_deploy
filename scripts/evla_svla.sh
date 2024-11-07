ckpts=(
  # llama_tokenizer # NOTE: 6, correct traj, hard to grsp objs
  # llama_tokenizer_140000 # NOTE:7, correct traj, grsp objs, long time, put fail
  # llama_tokenizer_160000 # NOTE: 7.5, correct traj, grsp objs sucess, long time, put fail
  #
  # 2024-09-20_11-08-30_simpler_env_llama2_7b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node2_gpu16_checkpoint-100000 #NOTE: 7.25, correct traj, grsping objs, long time, put fail, try manitime
  # 2024-09-20_11-08-30_simpler_env_llama2_7b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node2_gpu16_checkpoint-120000 #NOTE: 7.25, correct traj, grsping objs, oppsite grasiping 1 -> 1, long time, put fail, try manitime
  # 2024-09-20_11-08-30_simpler_env_llama2_7b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node2_gpu16_checkpoint-80000
  # 2024-09-29_19-34-12_simpler_env_llama3_2_3b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-220000 #NOTE: 5.5, correct traj, grsp wrong objs, long time, put fail
  # 2024-10-04_23-28-26_simpler_env_llama2_7b_siglip_dino_dpt_b_zero3_tf32_warmup_005_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000
  # 2024-10-04_23-29-18_simpler_env_llama2_7b_siglip_dino_dpt_b_frezdpt_zero3_tf32_warmup_005_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # NOTE: 8.0, correct traj, grsp objs, long time, put succ
  # 2024-10-05_11-01-21_simpler_env_llama2_7b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-160000 # NOTE: 6.0, correct traj, grasp none objs
  # 2024-10-05_11-01-21_simpler_env_llama2_7b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000 # NOTE: 6.2, correct traj, grasp obj faile
  # 2024-10-06_21-58-42_multi_frame_llama2_7b_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node2_gpu8_checkpoint-60000-temp-2_5 # NOTE: 5.5, wrong traj
  # 2024-10-17_10-51-57_simpler_env_spatial_llama3_2_3b_siglip_dino_zero1_wo_spatial_tf32_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 5.2 , very wrong traj
  # 2024-10-19_13-39-10_simpler_env_lmhead_lama3_2_3b_siglip_dino_zero3_obs24_wospatial_wo_gaussian_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # NOTE: 5.4, wrong traj
  # 2024-10-19_14-32-11_simpler_env_lmhead_lama3_2_3b_siglip_dino_zero3_obs24_wospatial_un_tie_w_gaussian_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # NOTE: 6.0, not so good traj
  # 2024-10-19_15-27-09_simpler_env_lmhead_lama3_2_3b_siglip_dino_zero3_obs24_spatial_un_tie_w_gaussian_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # NOTE: 5.3, wrong traj
  # 2024-10-19_15-27-09_simpler_env_lmhead_lama3_2_3b_siglip_dino_zero3_obs24_spatial_un_tie_w_gaussian_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # NOTE: 5.3, wrong traj
  # 2024-10-22_17-43-54_simpler_env_zoe_lama3_2_3b_siglip_dino_zero3_obs24_spatial_tie_gaussian_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # NOTE: 5.3, wrong traj
  # 2024-10-23_13-14-57_simpler_env_lama3_2_3b_siglip_dino_zero3_obs24_spatial_untie_gauss8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # NOTE: 5.4, wrong traj
  # 2024-10-23_14-49-50_simpler_env_lama3_2_3b_dino_zero3_obs24_spatial_untie_gauss1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 #NOTE: 5.3, wrong traj
  # 2024-10-23_14-49-50_simpler_env_lama3_2_3b_dino_zero3_obs24_spatial_untie_gauss8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # NOTE: 5.6, wrong traj repeat
  # 2024-10-23_21-15-12_simpler_env_lama3_2_1b_dino_zero3_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # NOTE: 5.5, wrong traj, wrong obj grasped
  # 2024-10-24_22-49-28_simpler_env_lama3_2_3b_dino_zoe_zero1_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # NOTE: 5.4, wrong traj
  # 2024-10-24_22-49-28_simpler_env_lama3_2_3b_dino_zoe_zero1_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-50000 # NOTE:5.4, wrong traj
  # 2024-10-24_22-54-22_simpler_env_lama3_2_3b_dino_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # NOTE:5.4, wrong traj
  # 2024-10-24_22-54-22_simpler_env_lama3_2_3b_dino_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-50000 # NOTE:5.4, wrong traj
  2024-10-25_17-07-36_simpler_env_lama3_2_3b_dino_zoe_zero1_obs24_spatial_untie_gaussN8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-75000 #NOTE: 7.0 correct traj, grasped wrong objs
  # 2024-10-26_23-41-29_simpler_env_unicam_mlp_lama3_2_3b_dino_zoe_zero1_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 1, 0.5, 0
  # 2024-10-27_09-42-00_simpler_env_unicam_mlp_lama3_2_3b_dino_zoe_zero1_obs24_spatial_untie_gaussN8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 0, 0, 0
  # 2024-10-30_23-59-48_simpler_env_pe3dt_lama3_2_3b_siglip_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node2_gpu16_checkpoint-20000 #0, 0, 0
  # 2024-10-31_00-11-44_simpler_env_pe_lama3_2_3b_siglip_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-20000 # 0 , 0 , 0
  # 2024-10-31_00-34-19_simpler_env_dinoL_lama3_2_3b_dino_zero3_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 0, 0,0
  # 2024-11-05_10-43-55_simpler_env_lama3_2_3b_dinol_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_nf4_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 #0, 0, 0
  # 2024-11-06_10-50-23_simpler_env_lama3_2_3b_dinol_zero1_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000
  # 2024-11-06_11-36-45_oxe_spatial_vla_lama3_2_3b_dinol_fast_zero1_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node8_gpu64_checkpoint-40000
  # 23-34-19_simpler_env_internlm1.8b_vit300m_zero1_tf32_decay0.0_warmup0.005_linear_lr2e-5_bs32_ga1_node4_gpu32
)

for ckpt in ${ckpts[@]}; do
  python bridge/run_bridgev2_eval.py \
    --model_family openvla \
    --pretrained_checkpoint ../ckpts/pretrained/$ckpt
done
