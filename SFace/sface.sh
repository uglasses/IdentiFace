# Run from repo root.
# python3 SFace/sface_pair_cosine_similarity.py \
#   --img1 path/to/query.png \
#   --img2 path/to/gallery_image.png \
#   --device cuda
python3 SFace/sface_faiss_1n.py \
  --query path/to/query.png \
  --gallery_dir path/to/gallery_dir \
  --top_k 10 \
  --device cuda
