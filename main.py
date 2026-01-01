
from Network import *
from Environment import *
from State import State
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import loadPrcFileData
import matplotlib.pyplot as plt
import numpy as np
print("something")
loadPrcFileData("", "win-size 480 360")#set window size

drone=Drone()
engine=Engine(drone,debugCam=False,sensor_range=50)

scene =Scene()
testModule= Module()

scene.setStartPos(np.array([0,-5,0]))
#drone.setPos(np.array([0,-5,0]))
#wall=Wall(np.array([10,0,0]))
scene.setStartRot(np.array([0,0,0]))
#drone.setRot(np.array([0,0,0]))

wall=Wall(np.array([10,100,1]))
wall.setPos(np.array([-1,0,-2]))
scene.addWall(wall)

wall=Wall(np.array([10,100,1]))
wall.setPos(np.array([-1,0,9]))
scene.addWall(wall)

wall=Wall(np.array([1,100,10]))
wall.setPos(np.array([-2,0,-1]))
scene.addWall(wall)

wall=Wall(np.array([1,100,10]))
wall.setPos(np.array([9,0,-1]))
scene.addWall(wall)
#engine.addWall(wall,True)

wall=Wall(np.array([10,1,5]))
wall.setPos(np.array([-1,5,-1]))
scene.addWall(wall)

wall=Wall(np.array([5,1,10]))
wall.setPos(np.array([-1,10,-1]))
scene.addWall(wall)

obs=mover(np.array([1,1,1]))
obs.setPos(np.array([10,0,0]))
#scene.addObstacle(obs)

scene.setGoal(np.array([0,15,0]))
#engine.addObstacle(obs,True)
scene1=Scene()

scene1.setGoal(np.array([10,10,10]))
f=open(r"test.txt")
testModule._unpackJson(json.loads(f.read()))
f.close()
scene1.addModule(testModule)


#drone.setPos(np.array([0,-15,0]))

#engine.addWall(wall)
#engine.addObstacle(obstacle)

#engine.registerWalls()
#engine.registerObstacles()

engine.loadScene(scene,debug=False)
#engine.unloadScene()
engine.drone.setPos(np.array([0,0,4]))
engine.renderFrame()


stop =False
#engine.addObstacle(obs,showCollider=True)
#engine.setGoalRender(True) <- rendering the goal is now off by defult for preformance reasons
undoDelay=False
engine.addRuler(20)

SENSOR_DATA_SIZE=[40,30]

brain=drone_brain(0,0.1,0.1,0.01,input_size=SENSOR_DATA_SIZE)

while not stop:
 engine.tick()
 engine.updateCam()
 taskMgr.step()
 engine.renderFrame()


 has_coll=engine.cheackCollision()
 if(has_coll):
     print("crash!")

 if engine.getGoalDist() <=0:
     print("Success!")
     stop=True

 #generate state object from game data
 currState=State(engine.getDepthBuffer(*SENSOR_DATA_SIZE),has_coll,engine.getGoalDist() <=0,engine.getGoalDist(),engine.getGoalDistFromStart())

 #tmp=simulate_network(engine) #network interface here
 tmp=brain.act(currState).detach().numpy()[0]#<-remove indexing after output dim is fixed
 print(tmp.shape)
 if(startMover(engine)):
     #engine.addObstacle(obs,showCollider=True)
     plt.imshow(engine.getDepthBuffer(80,60),cmap="Greys")
     plt.show()
 if(devTools(engine)):
     engine.setDevTools(True)

 if(saveScene(engine)):
     dat=engine.scene.saveAsModule()
     f=open(r"tmp.txt","w")
     f.write(json.dumps(dat))
     f.close()

 if(undo(engine)and not undoDelay):
     engine.delLast()
     undoDelay=True
 undoDelay=undo(engine)


 engine.drone.move(np.clip(tmp[:3],-0.1,0.1))
 engine.drone.rotate(np.clip(tmp[3:],-0.5,0.5))

 time.sleep(0.016666)



engine.destroy() #DO NOT REMOVE prevents spyder from entering a loop

