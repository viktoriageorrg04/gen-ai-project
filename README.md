Env setup
------------------
- that's how you set it up manually (it's probably easier to just run the notebook directly and it sets it up for you automatically with the packages requirements files; that's just as an alsternative)...
1) Create a venv inside `code/`:
   - `cd code`
   - `python -m venv .venv`
2) Install deps:
   - `.\.venv\Scripts\python -m pip install -r requirements_min.txt`
3) (Optional) Register a Jupyter kernel:
   - `.\.venv\Scripts\python -m ipykernel install --user --name genai-brand --display-name "GenAI Brand"`
4) Open `code/main_interface.ipynb` and run cells top-to-bottom.

Notes
- Backend (gen): diffusers only.
- Env files live in `code/.env` (auto-created from `code/.env.example` by the notebook).
- Outputs saved in `outputs/`.
- Added some error handling so the notebook might throw some here and there; before proceeding with generating the manifest, you should set `proceed = True` to continue as I couldn't figure out how to have it displayed by default.

Running the Notebook
--------------------
1) Open `code/main_interface.ipynb`.
2) Select the `GenAI Brand` kernel (or the `.venv` python).
3) Run the cells as follows:
   - Environment + imports
   - System A (manifest)
   - System B (generation)
   - System C (CLIP evaluation)
   - Improved Generation Experiments
4) Save the notebook after runs if you want to keep the outputs.