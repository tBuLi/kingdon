.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/tbuli/kingdon/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Kingdon could always use more documentation, whether as part of the
official Kingdon docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/tbuli/kingdon/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `kingdon` for local development.

1. Fork the `kingdon` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/kingdon.git

3. Set up your fork for local development with `uv <https://docs.astral.sh/uv/>`_, which creates a virtualenv in ``.venv`` with kingdon and its test dependencies installed::

    $ cd kingdon/
    $ uv sync --group test

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that the linter and the tests pass::

    $ uv run --group dev ruff check
    $ uv run pytest

   To also run the array tests against torch and jax, add their group::

    $ uv run --group arrays pytest

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.10 up to 3.14. Check
   https://github.com/tBuLi/kingdon/actions
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ uv run pytest tests/test_kingdon.py


Deploying
---------

A reminder for the maintainers on how to deploy. ``master`` is protected, so the version bump goes through a pull request, and the release is triggered by tagging the merged commit.

1. On a branch from ``master``, bump the version and add a section to HISTORY.rst::

    $ git worktree add ../release/X.Y.Z -b release/X.Y.Z master
    $ cd ../release/X.Y.Z
    $ uv version --bump minor  # possible: major / minor / patch, add --dry-run for a dry run.
    $ git commit -am "Release $(uv version --short)"
    $ git push -u origin release/X.Y.Z
    $ gh pr create --base master --title "Release $(uv version --short)"

2. Once CI passes, merge the pull request, then tag the merged ``master`` and push the tag::

    $ cd ../../master
    $ git pull
    $ git tag v$(uv version --short)
    $ git push origin v$(uv version --short)

   Pushing the tag makes GitHub Actions build kingdon and publish it to PyPI. The workflow refuses to publish if the tag does not match the version in pyproject.toml.

3. Once the release is on PyPI, create the GitHub release from the tag::

    $ gh release create v$(uv version --short) --verify-tag --notes-file release_notes.md
