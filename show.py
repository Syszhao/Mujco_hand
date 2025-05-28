import mujoco.viewer
import  mujoco

model=mujoco.MjModel.from_xml_path('rm_75_6f_description.xml')
mujoco.viewer.launch(model)     