import mujoco_py
from mujoco_py import mjviewer, mjcore
from mujoco_py import mjtypes
from mujoco_py import glfw
import numpy as np
from math import *
import random
import copy

from os.path import dirname, abspath


class MujocoEnv():   
    def __init__(self):
        self.xml_path = dirname(dirname(abspath(__file__))) + '/simulation/models/table_setup.xml' 
        self.model = mjcore.MjModel(self.xml_path)
        self.dt = self.model.opt.timestep;
        #self.action_space = spaces.Box(self.lower, self.upper)
        self.metadata = {'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : int(np.round(1.0 / self.dt))}

    def viewerSetup(self):
        self.width  = 640
        self.height = 480
        self.viewer = mjviewer.MjViewer(visible=True,
                                        init_width=self.width,
                                        init_height=self.height)

        #self.viewer.cam.trackbodyid = 0 #2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[0] = 0 #0.8
        self.viewer.cam.lookat[1] = 0.5 #0.8
        self.viewer.cam.lookat[2] = 0.1 #0.8
        self.viewer.cam.elevation = 160
        self.viewer.cam.azimuth = 100
        #self.viewer.cam.pose = 
        self.viewer.cam.camid = -3

        self.viewer.start()
        self.viewer.set_model(self.model)

        #(data, width, height) = self.viewer.get_image()

    def viewerEnd(self):
        self.viewer.finish()
        self.viewer = None

    def viewerStart(self):
        if self.viewer is None:
            self.viewerSetup()
        return self.viewer       

    def viewerRender(self):
 
        self.viewerStart().loop_once()
                                               
    def resetModel(self):
        self.model.resetData()
        ob = self.resetModel()
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewerSetup()
        return ob
          
    def getPos(self):
        return self.model.data.qpos.flat[0:3]

    def getOri(self):
        return self.model.data.qpos.flat[3:7]

    def getVel(self):
        return self.model.data.qvel.flat[0:3]

    def getOmg(self):
        return self.model.data.qvel.flat[3:6]
        
    def setPos(self, pos):
        q = copy.deepcopy(self.model.data.qpos)
        q[0:3] = pos
        self.model.data.qpos = q

    def setVel(self, vel):
        dq = copy.deepcopy(self.model.data.qvel)
        dq[0:3] = vel
        self.model.data.qvel = dq

    def setOmg(self, omg):
        dq = copy.deepcopy(self.model.data.qvel)
        dq[3:6] = omg
        self.model.data.qvel = dq
        
    def setControl(self, ctrl):
        self.model.data.ctrl = ctrl

    def step(self):
        self.model.step()  

    def resetBox(self):
        self.setPos(np.array([0,0,0]))
        self.setOri(np.array([1,0,0,0]))
        self.setVel(np.zeros([0,0,0]))
        self.setOmg(np.zeros([0,0,0]))
        self.model.qfrc_applied = np.zeros((self.model.nv,1))

    def applyFTOnObj(self, action_direction):

        # site_names = self.model.site_names

        # if not site_names:

        #     print "No sites found to apply inputs"
        #     raise ValueError


        # point1_index = random.randint(0, len(site_names)-1)
        # point2_index = random.randint(0, len(site_names)-1)

        # point1 = self.model.site_pose(site_names[point1_index])[0]
        # point2 = self.model.site_pose(site_names[point2_index])[0]
        # com    = self.model.data.xipos[1]

        # f_direction1 = (com-point1)/np.linalg.norm(com-point1)
        # f_direction2 = (com-point2)/np.linalg.norm(com-point2)

        # force1 = 500.*f_direction1# magnitude times direction
        # force2 = 500.*f_direction2#

        # torque = np.random.randn(3)

        force = 5*np.hstack([action_direction, 0])

        qfrc_target = self.model.applyFT(point=np.array([0, 0, 0]), 
                                   force=force, 
                                   torque=np.zeros(3), body_name='Box')
        
        self.model.data.qfrc_applied = qfrc_target


class PushBox(MujocoEnv):

    def __init__(self):
        self.box = MujocoEnv()
        self.box.viewerSetup()
        self.box.viewerStart()
        self.collected_rewards = []
        self.directions = [[1,0], [0,1], [-1,0],[0,-1], [0., 0.]]
        self.num_actions      = len(self.directions)
        self.goal = np.array([1,0,0])
        self.final_reward = 100
        self.thresh = 0.01
        #size of observation, 3 position and 3 velocity
        self.observation_size = 3 + 3 

    def observe(self):

        observation = np.hstack([self.box.getPos(), self.box.getVel()])

        return observation

    def check_reached_goal(self, pos):

        displacement = abs(self.goal - pos)

        if (displacement[0] < self.thresh) and\
           (displacement[1] < self.thresh) and\
           (displacement[2] < self.thresh):
            
            return self.final_reward

        else:

            return 0.


    def collect_reward(self):

        curr_pos = self.box.getPos()

        total_reward = -np.linalg.norm(curr_pos-self.goal) + self.check_reached_goal(curr_pos)

        self.collected_rewards.append(total_reward)
        
        return total_reward

    def perform_action(self, action):
        self.box.applyFTOnObj(action)

    def step(self, dt):
        num_steps = int(round(self.box.dt/dt))
        
        if num_steps <= 0:
            num_steps = 1

        for _ in range(num_steps):
            self.box.step()

    def to_html(self, info=[]):
        pass



if __name__ == "__main__":
    
    myBox = PushBox()

    while True:
        myBox.box.viewerRender()
        myBox.step(0.1)
        # myBox.resetBox()
        # myBox.applyFTOnObj()
        # for j in range(100):
        #     myBox.viewerRender()
        #     myBox.model.step()
    myBox.box.viewerEnd()