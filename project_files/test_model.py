
from simple_Network_sanitiy import *
from Environment import *
from State import State
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import loadPrcFileData
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress, MofNCompleteColumn , TextColumn, BarColumn
print("sanity check")
loadPrcFileData("", "win-size 600 600")#set window size

drone=Drone()
engine=Engine(drone,debugCam=False,sensor_range=50,Fov=60)

LEVEL_LENGTH=1

scene1,modules=compile_random_scene("level_modules",LEVEL_LENGTH)
scene1,modules=compile_scene(["L_top_right.txt"],"level_modules")

engine.renderFrame()


SENSOR_DATA_SIZE=(600,600)

brain=drone_brain(explore_factor=0,explore_decay=0,explore_min=0,lr=0,input_size=SENSOR_DATA_SIZE,near_dist=engine.getNear(),far_dist=50)
best_reward=2

torch.autograd.set_detect_anomaly(True) #enable traceback for torch backprop errors

start_epoch=brain.load(r"D:\eyals_staff\final proj\Env V2\avgpool 24\model")



losses=[]

def gameLoop(engine,brain,epoch):
    engine.loadScene(scene1,debug=False)
    start_pos=np.random.uniform(low=3,high=7,size=(3))
    engine.drone.setPos(start_pos)
    stop = False
    undoDelay=False
    generate_new=False
    while not stop:
        #update everything
        engine.tick()
        engine.updateCam()
        taskMgr.step()#panda3d's tick function
        engine.renderFrame()



        has_coll=engine.cheackCollision()
        if(has_coll):
            #print("crash!")
            stop=True

        if engine.getGoalDist() <=0:
            print("Success!")
            stop=True
            generate_new=True

        #generate state object from game data
        currState=State(engine.getDepthBuffer(*SENSOR_DATA_SIZE),has_coll,engine.getGoalDist() <=0,engine.getGoalDist(),engine.getGoalDistFromStart())
        reward=brain.analize_stateV6(currState)
        #best_reward=max(best_reward,reward)
       
        tmp,raw_pred=brain.act(currState,False) #predict best move
        raw_pred=raw_pred.detach()

        tmp=brain.classDecodeLRFlipped(np.argmax(raw_pred))
        #tmp=simulate_network(engine) #network interface here
        #print(raw_pred)
        tmp=tmp.squeeze()
        tmp[2]=-tmp[2]
        #dev tools
        #do NOT use while training!
        if(startMover(engine)): 
            #engine.addObstacle(obs,showCollider=True)
            plt.imshow(currState.sensorDat,cmap="Greys")
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

        #preform movement
        engine.drone.move(np.clip(tmp[:3],-0.1,0.1))
        engine.drone.rotate(np.clip(tmp[3:],-0.5,0.5))
        engine.drone.rotation=np.clip(engine.drone.rotation,-50,50)
        engine.drone.move(np.array([0,0.05,0]))


        bar.update(bar_task,advance=0,best_reward=round(best_reward,2),curr_reward=round(reward,2))

        #time.sleep(0.016666)
    brain.noLearn()
    engine.unloadScene()
    return generate_new
brain.explore_factor=0
epochs=100
#fancy progress bar 
best_reward=2
levels_beaten=0
tries=0
avg_tries=0
best_score_display=TextColumn("Best Reward: {task.fields[best_reward]}")
success_rate_display=TextColumn("success rate: {task.fields[success_rate]}")
current_score_display=TextColumn("Current Reward: {task.fields[curr_reward]}")
levels_beaten_display=TextColumn("levels beaten: {task.fields[levels_beaten]}")
avg_tries_display=TextColumn("avg tries: {task.fields[avg_tries]}")

bar=Progress(TextColumn("[progress.description]{task.description}"),BarColumn(),MofNCompleteColumn(),best_score_display,success_rate_display,current_score_display,levels_beaten_display,avg_tries_display)
bar_task=bar.add_task("running...",total=epochs,best_reward="N/A",success_rate="N/A",curr_reward="N/A",levels_beaten=0,avg_tries=0)
bar.start()
#training loop
for epoch in range(epochs):
    generate_new=gameLoop(engine,brain,epochs-epoch)
    best_reward=min(best_reward,min(brain.lifetime_rewards[-1]))
    
        #scene1,modules=compile_random_scene("level_modules",LEVEL_LENGTH)
    scene1,modules=compile_scene(["half_top.txt"],"level_modules")
    best_reward=0
    tries+=1
    if generate_new:
        levels_beaten+=1
    bar.update(bar_task,advance=1,best_reward=round(best_reward,2),curr_reward=brain.lifetime_rewards[-1],success_rate=("N/A" if levels_beaten==0 else ("1/"+str(tries/levels_beaten))),levels_beaten=levels_beaten,avg_tries=avg_tries)
#plot rewards graph
rewards=[]
for x in brain.lifetime_rewards:
    rewards+=x
plt.plot(rewards)
plt.show()

plt.plot(losses)
plt.show()

#plot avarege reward per epoch
avgs=[]
for x in brain.lifetime_rewards:
    avgs.append(sum(x)/len(x))
plt.plot(avgs)
plt.show()

engine.destroy() #DO NOT REMOVE prevents spyder from entering a loop



