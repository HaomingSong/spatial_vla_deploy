ckpts=(
  # llama_tokenizer # NOTE: 6, correct traj, hard to grsp objs
  # llama_tokenizer_140000 # NOTE:7, correct traj, grsp objs, long time, put fail
  # 4090_pretrained/llama_tokenizer_160000 # NOTE: 7.5, correct traj, grsp objs sucess, long time, put fail
  #
  # 2024-09-20_11-08-30_simpler_env_llama2_7b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node2_gpu16_checkpoint-100000 #NOTE: 7.25, correct traj, grsping objs, long time, put fail, try manitime
  # 2024-09-20_11-08-30_simpler_env_llama2_7b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node2_gpu16_checkpoint-120000 #NOTE: 7.25, correct traj, grsping objs, oppsite grasiping 1 -> 1, long time, put fail, try manitime
  # 2024-09-20_11-08-30_simpler_env_llama2_7b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node2_gpu16_checkpoint-80000
  # 2024-09-29_19-34-12_simpler_env_llama3_2_3b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-220000 #NOTE: 5.5, correct traj, grsp wrong objs, long time, put fail
  # 2024-10-04_23-28-26_simpler_env_llama2_7b_siglip_dino_dpt_b_zero3_tf32_warmup_005_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000
  # 4090_pretrained/2024-10-04_23-29-18_simpler_env_llama2_7b_siglip_dino_dpt_b_frezdpt_zero3_tf32_warmup_005_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # NOTE: 8.0, correct traj, grsp objs, long time, put succ
  # 2024-10-05_11-01-21_simpler_env_llama2_7b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-160000 # NOTE: 6.0, correct traj, grasp none objs
  # 2024-10-05_11-01-21_simpler_env_llama2_7b_siglip_dino_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000 # NOTE: 6.2, correct traj, grasp obj faile
  # 2024-10-06_21-58-42_multi_frame_llama2_7b_vit300m_zero1_tf32_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node2_gpu8_checkpoint-60000-temp-2_5 # NOTE: 5.5, wrong traj,
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
  # 2024-10-25_17-07-36_simpler_env_lama3_2_3b_dino_zoe_zero1_obs24_spatial_untie_gaussN8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-75000 #NOTE: 7.0 correct traj, grasped wrong objs
  # 2024-10-26_23-41-29_simpler_env_unicam_mlp_lama3_2_3b_dino_zoe_zero1_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 1, 0.5, 0
  # 2024-10-27_09-42-00_simpler_env_unicam_mlp_lama3_2_3b_dino_zoe_zero1_obs24_spatial_untie_gaussN8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 0, 0, 0
  # 2024-10-30_23-59-48_simpler_env_pe3dt_lama3_2_3b_siglip_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node2_gpu16_checkpoint-20000 #0, 0, 0
  # 2024-10-31_00-11-44_simpler_env_pe_lama3_2_3b_siglip_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-20000 # 0 , 0 , 0
  # 2024-10-31_00-34-19_simpler_env_dinoL_lama3_2_3b_dino_zero3_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 0, 0,0
  # 2024-11-05_10-43-55_simpler_env_lama3_2_3b_dinol_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_nf4_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 #0, 0, 0
  # 2024-11-06_10-50-23_simpler_env_lama3_2_3b_dinol_zero1_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000
  # 2024-11-06_11-36-45_oxe_spatial_vla_lama3_2_3b_dinol_fast_zero1_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node8_gpu64_checkpoint-40000
  # 23-34-19_simpler_env_internlm1.8b_vit300m_zero1_tf32_decay0.0_warmup0.005_linear_lr2e-5_bs32_ga1_node4_gpu32
  # 2024-11-07_16-08-51_simpler_env_lama3_2_3b_dinob_zoe_zero1_obs24_spatial_untie_gaussN1026_nf8_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0, 0, 0
  # obs14
  # 2024-11-08_17-19-51_simpler_env_lama3_2_3b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0, 1, 1
  # 2024-11-08_17-20-35_simpler_env_lama3_2_3b_dinob_fast_zero1_obs12_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0, 0, 0
  # 2024-11-08_17-44-37_simpler_env_lama3_2_3b_dinob_fast_zero1_obs14_spatial_untie_gaussN8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0, 0.5, 0.5
  # 2024-11-08_17-26-43_simpler_env_lama3_2_3b_dinob_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-50000 # 0, 0.0, 0.5
  # siglip
  # 2024-11-08_21-46-19_simpler_env_lama3_2_3b_siglip_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-50000
  # 2024-11-08_17-26-43_simpler_env_lama3_2_3b_dinob_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-70000 # 0.75, 0, 0.75 HACK: not too bad,
  # 2024-11-08_17-41-20_simpler_env_lama3_2_3b_dino_siglip_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0.5, 0, 0.5
  # 2024-11-08_17-19-51_simpler_env_lama3_2_3b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000 # 0, 0, 0
  # 2024-11-08_17-20-35_simpler_env_lama3_2_3b_dinob_fast_zero1_obs12_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 0.5, 0.5, 0.5 HACK: obs12 is better than obs14?
  # 2024-11-08_17-44-37_simpler_env_lama3_2_3b_dinob_fast_zero1_obs14_spatial_untie_gaussN8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000 # 0, 0, 0
  # 2024-11-08_21-46-19_simpler_env_lama3_2_3b_siglip_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0, 0, 0
  # 2024-11-09_01-16-51_simpler_env_lama3_2_3b_dinob_zoe_zero1_obs14_spatial_untie_uniN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-50000 # 0.5, 0, 0
  # 2024-11-10
  # 2024-11-08_23-02-38_simpler_env_lama3_2_3b_dinob_fast_zero1_obs14_ds1_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0.75, 0.5, 0.5 # HACK:downsample ratio 1 performs better but significantly slower 2x
  # 2024-11-09_22-13-45_simpler_env_lama2_3b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 0, 0, 0 box
  # 2024-11-09_01-16-51_simpler_env_lama3_2_3b_dinob_zoe_zero1_obs14_spatial_untie_uniN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0, 0, 0
  # 2024-11-09_01-21-50_simpler_env_lama3_2_3b_dino_zoe_zero1_obs14_spatial_untie_linea256_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # 0, 0, 0 box
  # obs11
  # 2024-10-31_23-31-24_simpler_env_lama3_2_3b_dino_siglip_zoe_zero1_obs11_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 0,0,0
  # llama2

  # 2024-11-09_22-13-45_simpler_env_lama2_3b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # 0,0,0 early stop
  # 2024-11-09_21-42-33_simpler_env_lama2_7b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 0,0.5,0. smooth slow
  # 2024-11-09_21-43-08_simpler_env_lama2_7b_dinol_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 #1.0,1.0,1.0 twice #HACK: llama2 7b + dinol
  # 2024-11-09_21-46-44_simpler_env_lama2_7b_dinob_fast_zero1_obs14_spatial_untie_gaussN8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 #0.25,0.5,0.25
  #
  # 2024-11-09_22-22-04_simpler_env_lama2_3b_dinol_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-50000 #0,0,0 poor,
  # 2024-11-09_22-58-56_simpler_env_lama2_3b_dino_siglip_zero1_obs18_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 #0,0.5,0.5
  # 2024-11-08_17-41-20_simpler_env_lama3_2_3b_dino_siglip_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 #0.25,0.75,0.75
  # 2024-11-09_22-59-27_simpler_env_lama2_7b_dinob_siglip_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-30000 #0,0.25,0.5
  # 2024-11-11
  # 2024-11-09_22-13-45_simpler_env_lama2_3b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 #0.25,0,0, HACK: inaccuarte
  # 2024-11-09_21-42-33_simpler_env_lama2_7b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-70000 #0, 0.5, 0.5
  # 2024-11-09_21-46-44_simpler_env_lama2_7b_dinob_fast_zero1_obs14_spatial_untie_gaussN8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 #0,0,0 small taj
  # 2024-11-10_11-04-28_simpler_env_lama3_2_3b_dinob_fast_zero1_obs11_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # rude, 0, 0.5, 0.5
  # 2024-11-10_11-04-28_simpler_env_lama3_2_3b_dinob_fast_zero1_obs11_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # rude, 0, 0, 0
  # 2024-11-10_11-06-04_simpler_env_lama3_2_3b_dinob_fast_zero3_obs11_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0,0,0
  # 2024-11-10_11-06-04_simpler_env_lama3_2_3b_dinob_fast_zero3_obs11_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-70000 #0.,0,0
  # 2024-11-09_22-13-45_simpler_env_lama2_3b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-130000 #0,0.5,0.5 HACK: llama2 3b + dinob performs better than llama3 3b dinob
  # 2024-11-09_21-43-08_simpler_env_lama2_7b_dinol_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-70000 #gentle, sucess 1, 1, 1, twice HACK:
  # 2024-11-10_13-37-21_simpler_env_lama3_2_3b_dinob_epoch10_zero1_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-50000 #rude, 0.5, 0., 0
  # 2024-11-09_22-22-04_simpler_env_lama2_3b_dinol_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000 #better, 1, 1, 1, sucess in twice performs worse in putting,  🔥 HACK:
  # 2024-11-09_01-16-51_simpler_env_lama3_2_3b_dinob_zoe_zero1_obs14_spatial_untie_uniN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 #rude, 0.5, 0.5, 0.5
  # 2024-11-09_13-15-53_simpler_env_lama3_2_3b_dino_siglip_zoe_zero1_obs18_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # 0,0,0
  # 2024-11-09_22-58-56_simpler_env_lama2_3b_dino_siglip_zero1_obs18_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 #0,0.5,0.5 obs14
  # 2024-11-09_22-59-27_simpler_env_lama2_7b_dinob_siglip_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-50000 # stuck, 0,0,0
  #
  # NOTE: steps for llama3.2
  # 2024-11-08_17-19-51_simpler_env_lama3_2_3b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 1.0, 1, 0.8,twice HACK:
  # 2024-11-08_17-20-35_simpler_env_lama3_2_3b_dinob_fast_zero1_obs12_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 0,0.5, 0
  # 2024-11-10_11-04-28_simpler_env_lama3_2_3b_dinob_fast_zero1_obs11_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000 # 0,0,0
  # 2024-11-10_11-06-04_simpler_env_lama3_2_3b_dinob_fast_zero3_obs11_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # worse 0,0,0
  # llama3 2
  # 2024-11-09_22-13-45_simpler_env_lama2_3b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-110000 # 0,0,0 HACK: llama2 3b performs worse than llama3.2 3b
  # 2024-11-10_13-37-21_simpler_env_lama3_2_3b_dinob_epoch10_zero1_obs24_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000
  # llama2 7b
  # 2024-11-10_11-22-13_simpler_env_lama3_2_3b_dinob_fast_zero1_obs14_vis_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-50000 # HACK: 1,1,0.75 fial to grasp eggplat
  # 2024-11-09_21-42-33_simpler_env_lama2_7b_dinob_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000 # 0, 0, 0.75
  # 2024-11-09_21-43-08_simpler_env_lama2_7b_dinol_fast_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000# 0, 0, 0.75
  # 2024-11-09_21-46-44_simpler_env_lama2_7b_dinob_fast_zero1_obs14_spatial_untie_gaussN8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000 #0,0,0
  # 2024-11-09_22-22-04_simpler_env_lama2_3b_dinol_zoe_zero1_obs24_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 #0,0,1
  # 2024-11-12 vision freeze
  # 2024-11-10_11-22-13_simpler_env_lama3_2_3b_dinob_fast_zero1_obs14_vis_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-70000 # 1, 1, 1, twice, eggplate
  # 2024-11-08_21-46-19_simpler_env_lama3_2_3b_siglip_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-70000
  # 2024-11-13 dinol
  # 2024-11-12_02-04-35_simpler_env_lama3_2_3b_dinol_fast_zero1_obs14_vis_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000  # 1,1,1 prettyy good, after 3 self try, sucesss
  # 2024-11-12_02-04-35_simpler_env_lama3_2_3b_dinol_fast_zero1_obs14_vis_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000  # 1,1,1, eggplant sucess after 5 times trying
  # 2024-11-12_02-04-35_simpler_env_lama3_2_3b_dinol_fast_zero1_obs14_vis_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 0.5, 1,1 less effective

  # 2024-11-12_02-07-25_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_N1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000
  # 2024-11-12_02-04-35_simpler_env_lama3_2_3b_dinol_fast_zero1_obs14_vis_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000

  # 2024-11-14
  # 2024-11-12_02-07-25_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_N1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # perfect 1,1,1, both carrot and eggplant, agin it do not work in gentle
  # 2024-11-12_20-32-28_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs15_spatial_untie_N1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # perfect 1 1 1, both carrot and eggplant, event smotther than obs14

  # 2024-11-12_02-04-35_simpler_env_lama3_2_3b_dinol_fast_zero1_obs14_vis_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-160000 # 1,1,1 but slightly slower
  #
  # 2024-11-12_14-17-48_simpler_env_lama3_2_3b_siglip_zoe_zero1_obs14_vis_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000/
  # siglip
  # 2024-11-08_21-46-19_simpler_env_lama3_2_3b_siglip_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-70000 # 0,0,0 | 0.8, 0, 0
  # 2024-11-12_14-17-48_simpler_env_lama3_2_3b_siglip_zoe_zero1_obs14_vis_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-110000 # HACK: SIGLIP eggplat sucess in 2 times clamp, 0.8,0.5,0.5 | 0,0,0
  # 2024-11-12_14-17-48_simpler_env_lama3_2_3b_siglip_zoe_zero1_obs14_vis_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-90000/
  # 2024-11-15
  # 2024-11-12_02-07-25_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_N1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-160000 # 1,1,1, putting rude
  # 2024-11-12_20-32-28_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs15_spatial_untie_N1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-140000 # 1,1,1, grasping above alitle
  # 2024-11-12_20-32-28_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs15_spatial_untie_N1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-160000
  # 2024-11-17 dino-fast wo system prompts and wo spatial
  # 2024-11-14_13-59-49_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_uniN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # basically good, accurate and gentle, fail in grasiping eggplant
  # 2024-11-14_13-59-49_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_uniN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 1,1,1 do not reconginize eggplate, but sfter 2 times trying, it works
  # 2024-11-15_21-38-07_simpler_env_lama3_2_3b_dinol_vis_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # 1,1,1 do not regonize eggplat, but in single eggplate tesing it grasp
  # 2024-11-15_22-07-56_simpler_env_lama3_2_3b_dinol_vis_zero1_obs14_spatial_untie_wospatial_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # 0,1,1 do not regonize eggplat, while wo/spatial do not grasp instead it grasp basket, so i'm pretty sure that spatial embedding enhance the performance

  # 8196
  # 2024-11-17_11-52-08_simpler_env_lama3_2_3b_dinol_vis_zero1_obs14_spatial_untie_gaussN8194_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # 1,1,0.9 | 🤗 1,0.9,1

  # 1b
  # 2024-11-17_11-37-06_oxe_spatial_vla_lama3_2_1b_dinol_vis_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node4_gpu32_checkpoint-140000 # 0.9, 1, 0.9 | 0
  # 2024-11-17_11-37-06_oxe_spatial_vla_lama3_2_1b_dinol_vis_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node4_gpu32_checkpoint-240000 # border 0.5, 0, 0.5
  # 2024-11-17_11-37-06_oxe_spatial_vla_lama3_2_1b_dinol_vis_zero1_obs14_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node4_gpu32_checkpoint-268133 # 0.5, 0, 0.5

  # 3b
  # 2024-11-18_02_oxe_spatial_vla_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-80000 # 1,1,1 in twice | | 0, 1, 1
  # 2024-11-18_02_oxe_spatial_vla_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-120000 # 0,9, 1, 0.9 | 0, 0.9, 0.9
  # 2024-11-18_02_oxe_spatial_vla_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-178755 # 0.9, 0, 0.9 | 0, 0.9, 0.9

  # gauss x uniform
  # 2024-11-18_00-36-45_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 1,1,1 | 0, 1,1
  # 2024-11-18_00-39-54_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_uniN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 1,1,1 | 0.8, 1, 0
  #
  # 2024-11-18_14-22-07_simpler_env_lama3_2_3b_dinol_vis_zero1_obs14_spatial_untie_gaussN1026_spatbin_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # 1,1,1 | 0, 1, 1 perfectly smooth
  # 2024-11-25 pretained N8194
  # 2024-11-24_01_oxe_spatial_vla_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-40000
  # 2024-11-24_01_oxe_spatial_vla_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-60000
  # 2024-11-24_01_oxe_spatial_vla_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-80000
  # 2024-11-27
  # 2024-11-24_01_oxe_spatial_vla_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-100000 # 1,1,1 | 0
  # 2024-11-27_00-36-58_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_stage3_checkpoint-10000 # 0, 0, 1
  # 2024-11-24_18-42-53_simpler_env_lama3_2_3b_dinol_vis_zero1_obs11_spatial_untie_gaussN1026_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0, 1,1, | 0
  # stage3
  # 2024-11-27_08-27-25_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-6_bs32_ga1_node1_gpu8_stage3_checkpoint-10000
  # siglip
  # 2024-11-24_18-22-38_simpler_env_lama3_2_3b_dino_siglip_vis_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0.5, 0.5, 0.5 | 0
  # 2024-11-24_20-02-43_simpler_env_lama3_2_3b_siglip_zoe_zero1_obs14_vis_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-50000 # 1, 1, 1 | 0
  # stage3 8194
  # 2024-11-27_09-30-12_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-6_bs32_ga1_node1_gpu8_stage3_checkpoint-10000
  # 2024-11-28 stage3 N8194 ptm
  # 2024-11-27_01_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr1e-6_bs32_ga1_node1_gpu8_stage3_checkpoint-40000
  # 2024-11-24_01_oxe_spatial_vla_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-160000
  # 2024-11-29
  # 2024-11-27_01_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr1e-6_bs32_ga1_node1_gpu8_stage3_checkpoint-80000 # 1,0.5,0
  # 2024-11-27_01_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # 1,1,1 | 0
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0
  # 2024-11-28_01_oxe_spatial_vla_lama3_2_1b_vis_dinob_siglip_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node4_gpu32_checkpoint-40000
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zoe_eb_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 1, 0.5, 0 | 0
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zoe_pe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 0.5, 0.5, 0.5 | 1 🤗 it do relly know eggplat
  # 2024-11-27_01_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr1e-6_bs32_ga1_node1_gpu8_stage3_checkpoint-100000 # 1, 0.8, 1 | 0
  # 2024-11-27_01_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 1,1,1 | 1 🤗

  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zoe_eb_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 1, 1, 1 | 0
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zoe_pe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 1, 1, 1 | 0

  # 2024-11-28_01_oxe_spatial_vla_lama3_2_1b_vis_dinob_siglip_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node4_gpu32_checkpoint-60000 # 1,1,1 | 0
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinol_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 0.5,0.5,0.5 | 0
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # 1,1,1 | 0
  # 2024-11-27_01_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr1e-6_bs32_ga1_node1_gpu8_stage3_checkpoint-120000
  # 2024-11-27_01_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000

  # 2024-11-30
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinol_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 1,1,1 | 0 x smooth and accurate, roubust, know theeggplate sometimes
  # 2024-11-28_01_oxe_spatial_vla_lama3_2_1b_vis_dinob_siglip_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node4_gpu32_checkpoint-80000 # 0,0,0
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # 1,1,1 | 0, not accurate
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zoe_eb_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 0.8,1,0.8 | 0
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zoe_pe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 0.8,1,1 | 0
  # 2024-11-27_01_simpler_env_lama3_2_3b_vis_dinol_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-140000 # 0.8,1,1, | 0 x
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 1, 0.8,1 | 1 🤗 , the sota now!
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinol_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000
  #

  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinol_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # 1,1,1 | 0
  # 2024-11-28_01_oxe_spatial_vla_lama3_2_1b_vis_dinob_siglip_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node4_gpu32_checkpoint-100000 # 0.5, 0.5, 0.5 | 0
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zoe_eb_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # 1,1,1 | 0
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_zoe_pe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # 0.5 1 1 | 0
  # 2024-11-29_01_simpler_env_kuka_lama3_2_3b_vis_dinol_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # 1,1,1 | 0, general sometime

  # 2024-12-01
  # 2024-11-29_01_simpler_env_kuka_lama3_2_3b_vis_dinol_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # 0.5, 0.5, 0.5 | 0
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinob_siglip_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 1, 1 1 | 0 (left 1st
  # 2024-11-30_01_oxe_spatial_vla_kuka_lama3_2_1b_vis_dinol_zoe_ds1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node4_gpu32_epoch2_checkpoint-40000 #
  # 2024-11-28_01_simpler_env_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # 1,1,11 | 0
  #
  # 2024-12-02
  # 2024-12-01_01_oxe_spatial_vla_kuka_fmb_lama3_2_1b_vis_dinol_zoe_ds1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node4_gpu32_epoch2_checkpoint-40000

  # 2024-11-30_01_simpler_env_drop0_1_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0.5, 0, 0 | 0 put fail
  # 2024-11-30_01_simpler_env_drop0_2_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 1, 0, 0 | 0 put fail
  # 2024-11-30_01_simpler_env_drop0_4_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0,0,0 | 0
  # 2024-11-30_01_simpler_env_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0.9,1,1 | 0 x
  # 2024-12-01_01_simpler_env_drop0_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_wospatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 1,1,1 | 0, accurate
  # 2024-12-01_01_oxe_spatial_vla_kuka_fmb_lama3_2_1b_vis_dinol_zoe_ds1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node4_gpu32_epoch2_checkpoint-60000 #
  # 2024-12-01_01_simpler_env_cls_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-20000

  # 2024-12-03
  # 2024-12-01_01_simpler_env_cls_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # 0.8,1,1 | 1 🤗 do reall instruction following
  # 2024-12-01_01_simpler_env_drop0_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_wospatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000
  # 2024-12-01_01_oxe_spatial_vla_kuka_fmb_lama3_2_1b_vis_dinol_zoe_ds1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node4_gpu32_epoch2_checkpoint-80000
  # 2024-11-30_01_simpler_env_drop0_1_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000
  # 2024-11-30_01_simpler_env_drop0_2_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000
  # 2024-11-30_01_simpler_env_lama3_2_3b_vis_dinol_zoe_ds1_0_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000
  # 2024-12-05 paligemma
  # 2024-12-05_01_simpler_env_paligemma_3b_vis_zoe_zero1_obs14_wospatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-20000
  #
  # 2024-12-05_16_simpler_env_paligemma_3b_vis_zoe_zero1_obs14_spatial_tie_weight_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000
  # 2024-12-05_16_simpler_env_paligemma_3b_vis_zoe_zero1_obs14_wospatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000 # instruction following but performs worse
  # 2024-12-06_03_simpler_env_paligemma_3b_vis_zoe_zero1_obs1_25_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-20000
  # 2024-12-06_03_simpler_env_paligemma_3b_vis_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-20000
  # 2024-12-06_03_simpler_env_paligemma_3b_vis_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-20000
  # 2024-12-07
  # 2024-12-05_16_simpler_env_paligemma_3b_vis_zoe_zero1_obs14_wospatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000
  # 2024-12-06_03_simpler_env_paligemma_3b_vis_zoe_zero1_obs1_25_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000
  # 2024-12-06_03_simpler_env_paligemma_3b_frezvis_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # zero
  # 2024-12-06_03_simpler_env_paligemma_3b_vis_zoe_zero1_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000
  # 2024-12-06_03_simpler_env_paligemma_3b_vis_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000
  # 2024-12-06_09_oxe_spatial_vla_kuka_fmb_paligemma_3b_vis_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node4_gpu32_checkpoint-20000
  # 2024-12-09
  # 2024-12-07_19_simpler_env_paligemma_3b_vis_zoe_zero1_obs11_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0
  # 2024-12-07_19_simpler_env_paligemma_3b_vis_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000
  # 2024-12-07_19_simpler_env_paligemma_3b_vis_zoe_zero1_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000
  # 2024-12-08_00_oxe_spatial_vla_kuka_fmb_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-40000
  # 2024-12-08_00_simpler_env_paligemma_3b_vis_zoe_flash_attn_obs13_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-40000
  # 2024-12-08_17_simpler_env_paligemma1_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-20000
  # 2024-12-10
  # 2024-12-08_00_oxe_spatial_vla_kuka_fmb_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-80000 # 1,1,1
  # 2024-12-08_00_simpler_env_paligemma_3b_vis_zoe_flash_attn_obs13_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # not too bad
  # 2024-12-08_00_simpler_env_paligemma_3b_vis_zoe_flash_attn_obs13_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 # bad
  # 2024-12-08_17_simpler_env_paligemma1_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-80000 # bad
  #
  # 2024-12-08_00_oxe_spatial_vla_kuka_fmb_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-100000 # 1,1,1,
  # 2024-12-08_00_simpler_env_paligemma_3b_vis_zoe_flash_attn_obs13_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 #1,1,0
  # 2024-12-08_17_simpler_env_paligemma1_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-100000 #1,1,1 | 0
  # 2024-12-09_18_simpler_env_paligemma_3b_vis_zoe_flash_attn_obs1_25_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000
  # 2024-12-10_01_simpler_env_paligemma_3b_vis_wozoe_flash_attn_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 0,0,0
  # 2024-12-10_01_simpler_env_paligemma_3b_vis_zoe_flash_attn_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # 1,1,1, | 1 🤗 | corn fail
  # finetuning/spatial_vla/2024-12-11_01_spatialvla_all_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu4_checkpoint-20000
  # 2024-12-08_00_oxe_spatial_vla_kuka_fmb_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-120000

  # 2024-12-11_22_oxe_spatial_vla_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-40000 #1,0,0
  # 2024-12-12_10_oxe_spatial_vla_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node4_gpu32_checkpoint-40000 #1,1,1 | corn 1,0,1

  # ptm v1
  # 2024-12-14_02_oxe_spatial_vla_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-80000 # warn eggplate and carrot fuse, work well except corn
  # aug
  # 2024-12-15_01_simpler_env_paligemma_3b_vis_zoe_flash_attn_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-60000 # lower acc, data aug not work well? wait for longger ckpt
  #
  # 2024-12-14_02_oxe_spatial_vla_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-180000
  # 2024-12-14_02_oxe_spatial_vla_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-120000

  # zero-shot tesing
  # v1-1026
  # 2024-12-14_02_oxe_paligemma_3b_N1026_lr2e-5_bs32_gpu48_checkpoint_230770 # stage2
  # 2024-12-25_11_mix_sft5_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN1026_unicam_lr5e-6_bs32_ga1_node1_gpu8_checkpoint-64338 # stage 3 mix
  # 2024-12-24_20_mix_sft3_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN1026_unicam_lr2_5e-6_bs32_ga1_node1_gpu8_checkpoint-19694 # stage 3 mix
  # 2024-13-14_02_oxe_spatial_vla_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-180000

  # v2-8194
  # 2024-12-08_00_oxe_spatial_vla_kuka_fmb_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node6_gpu48_checkpoint-120000 # openx-- N8196 120k only, wo aug

  # v3 simplerenv dataset
  # ../ckpts/2024-12-26_12_simpler_env_paligemma3b_vis/_zoe_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # N1026
  # ../ckpts/2024-12-26_10_simpler_env_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # linear
  # ../ckpts/2024-12-27_05_simpler_env_paligemma3b_vis_zoe_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # N8194
  # ../ckpts/2024-12-26_19_simpler_env_paligemma3b_vis_zoe_obs14_wospatial_untie_gaussN1026_unicam_lr2e-5_bs32_ga1_node1_gpu8_checkpoint-120000 # wozoe

  # v4 pretrian 64gpus 8194 reso
  # ../pretrained/2025-01-05_09-12-37_oxe_spatial_vla_paligemma3b_zoe_gsN8194_gpu64-204k
  # ../pretrained/2025-01-14_12_mix_sft7_2025-01-05_09-12-37_oxe_spatial_vla_paligemma3b_zoe_gsN8194_gpu64-204k_lr1e-6_bs32_node1_gpu8_r0_a0_ep3_none_sft_sigm0_checkpoint-30000
  # ../pretrained/2025-01-14_12_mix_sft7_2025-01-05_09-12-37_oxe_spatial_vla_paligemma3b_zoe_gsN8194_gpu64-204k_lr1e-6_bs32_node1_gpu8_r0_a0_ep3_none_sft_sigm0_checkpoint-40000
  ../pretrained/2025-01-14_12_mix_sft7_2025-01-05_09-12-37_oxe_spatial_vla_paligemma3b_zoe_gsN8194_gpu64-204k_lr2_5e-6_bs32_node1_gpu8_r0_a0_ep3_none_sft_sigm0_checkpoint-30000
  # ../pretrained/2025-01-14_12_mix_sft7_2025-01-05_09-12-37_oxe_spatial_vla_paligemma3b_zoe_gsN8194_gpu64-204k_lr2_5e-6_bs32_node1_gpu8_r0_a0_ep3_none_sft_sigm0_checkpoint-40000

)

for ckpt in ${ckpts[@]}; do
  python bridge/run_bridgev2_eval_open_loop.py \
    --model_family openvla \
    --pretrained_checkpoint $ckpt \
    --ensemble_actions True \
    --action_horizon 4 \
    --max_steps 400
done
