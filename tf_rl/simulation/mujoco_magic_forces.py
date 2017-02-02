import mujoco_py
from mujoco_py import mjviewer, mjcore
from mujoco_py import mjtypes
from mujoco_py import glfw
import numpy as np
from math import *
import random
import copy
import matplotlib.pyplot as plt
from os.path import dirname, abspath
from config import MUJOCO_ENV

# plt.ion()

class MujocoEnv():   
    def __init__(self, actions, config=MUJOCO_ENV):
        self.config = config
        self.xml_path = config['model_name']
        self.model = mjcore.MjModel(self.xml_path)
        self.dt = self.model.opt.timestep;
        #self.action_space = spaces.Box(self.lower, self.upper)
        self.metadata = {'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : int(np.round(1.0 / self.dt))}
        self.actions = actions
        self.max_force = 2

    def viewerSetup(self):

        self.viewer = mjviewer.MjViewer(visible=True,
                                        init_width=self.config['image_width'],
                                        init_height=self.config['image_height'])

        self.viewer.start()
        self.viewer.set_model(self.model)

        # if 'camera_pos' in self.config:
        #     cam_pos = self.config['camera_pos']
        #     for i in range(3):
        #         self.viewer.cam.lookat[i] = cam_pos[i]
        #     self.viewer.cam.distance = cam_pos[3]
        #     self.viewer.cam.elevation = cam_pos[4]
        #     self.viewer.cam.azimuth = cam_pos[5]
        #     self.viewer.cam.trackbodyid = -1 

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
        return self.model.data.qpos[0:3].flatten()

    def getOri(self):
        return self.model.data.qpos[3:7].flatten()

    def getVel(self):
        return self.model.data.qvel[0:3].flatten()

    def getOmg(self):
        return self.model.data.qvel[3:6].flatten()
        
    def setPos(self, pos):
        q = copy.deepcopy(self.model.data.qpos)
        q[0:3] = pos
        self.model.data.qpos = q

    def setOri(self, ori):
        q = copy.deepcopy(self.model.data.qpos)
        q[3:7] = ori
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
        # print "external force \t", np.round(self.model.data.qfrc_applied.flatten(),3)
        self.model.step()
        self.viewerRender()  

    def resetBox(self):
        self.setPos(np.zeros((3,1)))
        self.setOri(np.zeros((4,1)))
        self.resetVel()
        self.model.qfrc_applied = np.zeros((self.model.nv,1))

    def resetVel(self):
        self.setVel(np.zeros((3,1)))
        self.setOmg(np.zeros((3,1)))
        self.model.step1()

    def applyFTOnObj(self, action_direction):

        self.resetBox()

        force = self.max_force*np.hstack([self.actions[action_direction], 0.])

        qfrc_target = self.model.applyFT(point=np.array([0., 0., 0.]), #-0.15 is the bottom of the box
                                         force=force, 
                                         torque=np.zeros(3), body_name='Box')
        
        self.model.data.qfrc_applied = qfrc_target#np.hstack([force1+force2, torque])[:,None]

        # print "force \t", force

        # print "qfrc_target \t", np.round(qfrc_target.flatten(),3)

        # print "qfrc applied \t", np.round(self.model.data.qfrc_applied.flatten(),3)


class PushBox(MujocoEnv):

    def __init__(self):
        self.collected_rewards = []
        self.directions   = [[1.,0.], [0.,1.], [-1.,0.],[0.,-1.], [-1.,-1.], [1.,1.], [1.,-1.], [-1.,1.], [0., 0.]]
        self.num_actions  = len(self.directions)
        self.box = MujocoEnv(actions=self.directions)
        self.box.viewerSetup()
        self.box.viewerStart()
        self.goal = np.array([1.35,1.35,0])
        self.final_reward = 1
        self.thresh = 0.3
        #size of observation, 3 position and 3 velocity
        self.observation_size = 3 + 3 

    def check_within_limits(self, pos):

        xlim = (pos[0] > 1.49) or (pos[0] < -1.49)
        ylim = (pos[1] > 1.49) or (pos[1] < -1.49)
        
        # print "pos \t", np.round(pos,3)

        zlim = (pos[2] > 0.05) 

        if zlim or xlim or ylim:
            return False
        else:
            return True


    def observe(self):

        # print "SOMEBODY CALLED ME observe"

        q = self.box.getPos()

        observation = np.hstack([q, self.box.getVel()])

        if not self.check_within_limits(q):
            # print "called reset"
            self.box.resetBox()

        return observation

    def check_reached_goal(self, pos):

        # print "From check_reached_goal \t", pos

        # print "From  goal \t", self.goal

        # print "Difference \t", self.goal-pos

        displacement = abs(self.goal - pos)

        # print "DISPLACEMENT \t", displacement
        # raw_input("Enter")

        if (displacement[0] < self.thresh) and\
           (displacement[1] < self.thresh):
            
            print "YAAAAAYYYYYYYYYYYYYY"
            return self.final_reward

        else:

            return 0.


    def collect_reward(self):

        # print "SOMEBODY CALLED ME collect_reward"

        curr_pos = self.box.getPos()

        # print "From  collect_reward, curr pos \t", curr_pos

        if not self.collected_rewards:
            present_total_reward = 0    
        else:
            present_total_reward = 0
            
        total_reward = self.check_reached_goal(curr_pos) + present_total_reward

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

    def plot_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.collected_rewards[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))

        # plt.plot(list(range(len(x))), x)
        # plt.pause(0.0001)
        # plt.show()

    # def to_html(self, info=[]):
        
    #     stats = stats[:]
    #     recent_reward = self.collected_rewards[-100:] + [0]
    #     objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in self.objects_eaten.items()])
    #     stats.extend(["reward       = %.1f" % (sum(recent_reward)/len(recent_reward),),
    #     ])

    #     scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(stats)))
    #     scene.add(svg.Rectangle((10, 10), self.size))

    #     offset = self.size[1] + 15
    #     for txt in stats:
    #         scene.add(svg.Text((10, offset + 20), txt, 15))
    #         offset += 20

    #     return scene

if __name__ == "__main__":
    
    myBox = PushBox()

    while True:
        myBox.box.viewerRender()
        myBox.perform_action(1)
        myBox.step(1)
        # myBox.resetBox()
        # myBox.applyFTOnObj()
        # for j in range(100):
        #     myBox.viewerRender()
        #     myBox.model.step()
    myBox.box.viewerEnd()