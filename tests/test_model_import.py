def test_import_model():
    import importlib
    m = importlib.import_module("hypermamba.model")
    assert hasattr(m, "main_model")
