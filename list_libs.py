"""
Script that lists all available Python libraries installed in the environment.
This script uses `pkg_resources` from the `setuptools` package to list the installed libraries.
"""

import pkg_resources

def list_installed_libraries():
    installed_libraries = [d.key for d in pkg_resources.working_set]
    for lib in sorted(installed_libraries):
        print(lib)

# Uncomment the following line to run the script
list_installed_libraries()
