from __future__ import annotations

import pathlib

from setuptools import find_packages, setup


def _load_python_pyproject() -> dict:
    root = pathlib.Path(__file__).resolve().parent
    pyproject_path = root / "python" / "pyproject.toml"
    try:
        import tomllib

        return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        import tomli

        return tomli.loads(pyproject_path.read_text(encoding="utf-8"))


def _as_console_scripts(project_scripts: dict[str, str] | None) -> list[str]:
    if not project_scripts:
        return []
    return [f"{name}={target}" for name, target in project_scripts.items()]


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    data = _load_python_pyproject()

    project = data.get("project", {})
    tool = data.get("tool", {})
    setuptools_cfg = tool.get("setuptools", {})

    packages_find_cfg = setuptools_cfg.get("packages", {}).get("find", {})
    exclude = packages_find_cfg.get("exclude", [])

    package_data_cfg = setuptools_cfg.get("package-data", {})
    sglang_package_data = package_data_cfg.get("sglang", [])

    readme_path = root / "README.md"
    long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    setup(
        name=project.get("name", "sglang"),
        version=project.get("version", "0.0.0"),
        description=project.get("description", ""),
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires=project.get("requires-python"),
        install_requires=project.get("dependencies", []),
        extras_require=project.get("optional-dependencies", {}),
        classifiers=project.get("classifiers", []),
        package_dir={"": "python"},
        packages=find_packages("python", exclude=exclude),
        package_data={"sglang": sglang_package_data},
        entry_points={
            "console_scripts": _as_console_scripts(project.get("scripts", {})),
        },
        license_files=["LICENSE"],
    )


if __name__ == "__main__":
    main()
