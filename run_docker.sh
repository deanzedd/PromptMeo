docker run --gpus all -it --rm \
  --ipc=host \
  --name promptmeo_runner \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -v $(pwd):/workspace/PromptMeo \
  promptmeo_rtx5090