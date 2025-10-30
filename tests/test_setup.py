import sys


def test_package_imports():
    """Test that the main package can be imported."""
    import deap_pettingzoo
    
    assert deap_pettingzoo.__version__ == "0.1.0"
    assert deap_pettingzoo.__author__ == "Alex"


def test_dependencies():
    """Test that all required dependencies are available."""
    import deap.base
    import deap.creator
    import deap.tools
    import pettingzoo
    import numpy
    import gymnasium
    
    # Basic sanity checks
    assert deap.base.Toolbox is not None
    assert deap.creator.create is not None
    assert deap.tools.selTournament is not None
    assert pettingzoo.__version__
    assert numpy.__version__
    assert gymnasium.__version__


def test_python_version():
    """Test that Python version is compatible."""
    assert sys.version_info >= (3, 8), "Python version >= 3.8 is required"


def test_submodule_imports():
    """Test that all submodules can be imported."""
    from deap_pettingzoo import genomes
    from deap_pettingzoo import policies
    from deap_pettingzoo import evaluators
    from deap_pettingzoo import environments
    from deap_pettingzoo import utils
    
    assert genomes is not None
    assert policies is not None
    assert evaluators is not None
    assert environments is not None
    assert utils is not None
