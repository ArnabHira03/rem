from rem.projects import resolve_project


def test_resolves_git_root_from_subdir(tmp_path):
    (tmp_path / "proj" / ".git").mkdir(parents=True)
    sub = tmp_path / "proj" / "src" / "a"
    sub.mkdir(parents=True)
    info = resolve_project(sub)
    assert info.root_path == str((tmp_path / "proj").resolve()) and info.name == "proj"
    assert info.git_remote is None


def test_falls_back_to_cwd_without_git(tmp_path):
    assert resolve_project(tmp_path).root_path == str(tmp_path.resolve())
