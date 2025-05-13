First, create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

You will also need to add a Hugging Face token and OpenAI key to `.env`. See
`.env.example`. For running on a cloud instance, you may want to set the
`HF_HOME` environment variable to a persistent location (defaults to
`~/.cache/huggingface`).

To reproduce the results in the paper, first run the input pipeline to generate
and compute the relevance scores for 5000 wikipedia articles:

```bash
./scripts/run_input_pipeline.sh 5000
```

Then, run the cells in `notebooks/results.ipynb` to generate the figures in the
paper. Note: you may need to rent a GPU instance with at least ~40GB of VRAM
to run the local model.
