# Compiling starling's Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/). To compile the docs, first ensure that Sphinx and the ReadTheDocs theme are installed.


```bash
conda install sphinx sphinx_rtd_theme 
```


Once installed, you can use the `Makefile` in this directory to compile static HTML pages by
```bash
make html
```

The compiled docs will be in the `_build` directory and can be viewed by opening `index.html` (which may itself be inside a directory called `html/` depending on what version of Sphinx is installed).


A configuration file for [Read The Docs](https://readthedocs.org/) (`readthedocs.yml`) is included in the top level of the repository. To use Read the Docs to host your documentation, go to https://readthedocs.org/ and connect this repository. You may need to change your default branch to `main` under Advanced Settings for the project.

Read the Docs installs the packages listed in `docs/requirements.txt` and then the package itself (`pip install .`, without extras). If `autodoc` needs a package that neither of these installs, either add it to `docs/requirements.txt` or add it to `autodoc_mock_imports` in `docs/conf.py`.

The API reference stubs in `docs/autosummary/` are committed, and `docs/conf.py` sets `autosummary_generate_overwrite = False`, so Sphinx only creates stubs that are missing and never refreshes existing ones. When modules, classes or functions are added, removed or renamed, regenerate the affected stubs (or delete them and rebuild) so the API pages don't go stale.
