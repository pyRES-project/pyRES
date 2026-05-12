"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""

from src.kernel.plot_result import  *



file_path_yaml = Path("Input/config.yaml")
output_dir=Path("Output")
simulation,all_components,rec_result, pros_result, rec_result_ec,pros_result_ec=run(file_path_yaml, base_path=Path(__file__).parent,output_dir=output_dir)
plot(simulation,all_components)

