
from simple_Network import *
from Environment import *
from State import State
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import loadPrcFileData
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress, MofNCompleteColumn , TextColumn, BarColumn
print("sanity check")
loadPrcFileData("", "win-size 480 360")#set window size

drone=Drone()
engine=Engine(drone,debugCam=False,sensor_range=50)

base_scene =Scene()
testModule= Module()

base_scene.setStartPos(np.array([0,-5,0]))
#drone.setPos(np.array([0,-5,0]))
#wall=Wall(np.array([10,0,0]))
base_scene.setStartRot(np.array([0,0,0]))
#drone.setRot(np.array([0,0,0]))

wall=Wall(np.array([10,100,1]))
wall.setPos(np.array([0,-10,0]))
base_scene.addWall(wall)

wall=Wall(np.array([10,100,1]))
wall.setPos(np.array([0,-10,10]))
base_scene.addWall(wall)

wall=Wall(np.array([1,100,10]))
wall.setPos(np.array([0,-10,0]))
base_scene.addWall(wall)

wall=Wall(np.array([1,100,10]))
wall.setPos(np.array([10,-10,0]))
base_scene.addWall(wall)
#engine.addWall(wall,True)

wall=Wall(np.array([10,1,10]))
wall.setPos(np.array([0,-7,0]))
base_scene.addWall(wall)

"""wall=Wall(np.array([10,1,5]))
wall.setPos(np.array([-1,5,-1]))
scene.addWall(wall)

wall=Wall(np.array([5,1,10]))
wall.setPos(np.array([-1,10,-1]))
scene.addWall(wall)

obs=mover(np.array([1,1,1]))
obs.setPos(np.array([10,0,0]))
#scene.addObstacle(obs)"""

base_scene.setGoal(np.array([0,40,0]))
f=open(r"level_modules/half_bottom.txt")
testModule._unpackJson(json.loads(f.read()))
f.close()
base_scene.addModule(testModule)

#engine.addObstacle(obs,True)
scene=Scene()

"""scene.setGoal(np.array([10,10,10]))
f=open(r"tmp.txt")
testModule._unpackJson(json.loads(f.read()))
f.close()
scene.addModule(testModule)"""


#drone.setPos(np.array([0,-15,0]))

#engine.addWall(wall)
#engine.addObstacle(obstacle)

#engine.registerWalls()
#engine.registerObstacles()



#engine.loadScene(scene1,debug=True)
#engine.unloadScene()

engine.renderFrame()


#engine.addObstacle(obs,showCollider=True)
#engine.setGoalRender(True) #<- rendering the goal is now off by defult for preformance reasons

engine.addRuler(20)

SENSOR_DATA_SIZE=[30,40]

#brain=drone_brain(explore_factor=0.4,explore_decay=0.01,explore_min=0.05,lr=0.1,input_size=SENSOR_DATA_SIZE)
#best_reward=-2
rando,names = compile_random_scene("level_modules",3)
#torch.autograd.set_detect_anomaly(True)
#engine.lazyLoadScene(base_scene,debug=False)
def gameLoop(engine,brain):
    engine.loadScene(rando,debug=False)
    engine.drone.setPos(np.array([4,0,4]))
    #engine.goal=np.array([0,100,0])
    #engine.drone.setPos(np.array([6,3,6]))
    stop = False
    undoDelay=False
    while not stop:
        engine.tick()
        engine.updateCam()
        taskMgr.step()
        engine.renderFrame()



        has_coll=engine.cheackCollision()
        if(has_coll):
            print("crash!")
            #stop=True

        if engine.getGoalDist() <=0:
            print("Success!")
            #stop=True

        #generate state object from game data
        currState=State(engine.getDepthBuffer(*SENSOR_DATA_SIZE),has_coll,engine.getGoalDist() <=0,engine.getGoalDist(),engine.getGoalDistFromStart())
        #reward=brain.analize_state(currState)
        #best_reward=max(best_reward,reward)
        tmp=simulate_network(engine) #network interface here
        #tmp=brain.act(currState)
        tmp=tmp.squeeze()
        if(startMover(engine)):
            #engine.addObstacle(obs,showCollider=True)
            plt.imshow(engine.getDepthBuffer(*SENSOR_DATA_SIZE),cmap="Greys")
            plt.show()
        if(devTools(engine)):
            engine.setDevTools(True)

        if(saveScene(engine)):
            dat=engine.scene.saveAsModule(offset=np.array([0,20,0]))
            f=open(r"mistake.txt","w")
            f.write(json.dumps(dat))
            f.close()

        if(undo(engine)and not undoDelay):
            engine.delLast()
            undoDelay=True
        undoDelay=undo(engine)


        engine.drone.move(np.clip(tmp[:3],-0.1,0.1))
        engine.drone.rotate(np.clip(tmp[3:],-0.5,0.5))

        time.sleep(0.016666)
    #brain.learn()
    engine.unloadScene()

gameLoop(engine,"dummy value")

engine.destroy() #DO NOT REMOVE prevents spyder from entering a loop

