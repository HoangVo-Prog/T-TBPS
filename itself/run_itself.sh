DATASET_NAME="RSTPReid"
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
--name PPL \
--output_dir 'ITSELF' \
--dataset_name $DATASET_NAME \
--loss_names 'tal+cid' \
--num_epoch 30 \
--lr_total_epochs 60 \
--return_all \
--topk_type 'custom' \
--modify_k \
--freeze_host --no_use_host_loss \
--finetune_clip /path/to/best.pth \
--target_enrichment --lambda_ret 0.5 \
--use_freeze_indices --recompute_interval -1 \
--residual_gate static --enrich_gamma 0.1 \
--topm_rank_space retrieval --enrichment_space retrieval \
--top_m 64 --mixer_depth 1 \
--mixer_hidden_rank 128 --mixer_hidden_readout 256 \
--extractor_mode horizontal --num_parts 12 \
--topm_rank_space hybrid_global_grab --topm_rank_lambda 0.4 \
--enrichment_space grab 