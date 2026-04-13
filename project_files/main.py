
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

torch.manual_seed(42)
np.random.seed(42)

drone=Drone()
FAR_DIST=50
engine=Engine(drone,debugCam=False,sensor_range=FAR_DIST,Fov=60)

LEVEL_LENGTH=1

scene1,modules=compile_random_scene("level_modules",LEVEL_LENGTH)#generate random level

#scene1,modules=compile_scene(["L_top_right.txt"],"level_modules")#generate fixed level - for debug

engine.renderFrame()


SENSOR_DATA_SIZE=(600,600)
#orgenaisation, to be changed for each run
RUN_IDETIFIER="avgpool 33"

NOTES="""
    basic network,\n
    explore factor 1, no decay\n
    the network is now as follows:\n
    x=self.avgpool(x_in)\n
    x=self.flatten(x)\n
    x=self.linear1(x)\n
    x=self.relu(x)
    x=self.linear2(x)\n
    x=self.relu(x)\n

    50 epochs\n
    full train using only L_top_right
    simplified reward
    avg pool kernal size 200x200
    
    
"""

#create files, comment if loading from file
os.mkdir(RUN_IDETIFIER)
f=open(RUN_IDETIFIER+r"/module_list","a")
dat=json.dumps(modules)
f.write(dat)
f.close()
f=open(RUN_IDETIFIER+r"/rewards.json","a")
f.write(json.dumps([]))
f.close()
f=open(RUN_IDETIFIER+r"/notes.txt","a")
f.write(NOTES)
f.close()
start_epoch=0

def save_rewards(path):
    rewards=[]
    for x in brain.lifetime_rewards:
        rewards+=x
    for x in brain.rewards:
        rewards+=x
    rewards=rewards.tolist()#no clue why rewards is a numpy array, but here is a fix regardless
    f=open(path,"w")
    f.write(json.dumps(rewards))
    f.close()

#create drone_brain object
brain=drone_brain(explore_factor=1,explore_decay=0,explore_min=0,lr=0.1,input_size=SENSOR_DATA_SIZE,far_dist=FAR_DIST,near_dist=engine.getNear())
best_reward=2

torch.autograd.set_detect_anomaly(True) #enable traceback for torch backprop errors

#start_epoch=brain.load(r"D:\eyals_staff\final proj\Env V2\basic CNN 7\model") #uncomment if loading from file


losses=[]

def gameLoop(engine,brain,epoch):
    #setup
    engine.loadScene(scene1,debug=False)
    start_pos=np.random.uniform(low=3,high=7,size=(3))
    engine.drone.setPos(start_pos)
    stop = False
    undoDelay=False
    generate_new=False
    dobreak=False
    override_counter=0
    timer=1e5
    #main loop
    while not stop and timer >=0:
        #update everything
        engine.tick()
        engine.updateCam()
        taskMgr.step()#panda3d's tick function
        engine.renderFrame()


        #cheack for collisions
        has_coll=engine.cheackCollision()
        if(has_coll):
            #print("crash!")
            stop=True
        #cheack for game end
        if engine.getGoalDist() <=0:
            print("Success!")
            stop=True
            generate_new=True

        #generate state object from game data
        currState=State(engine.getDepthBuffer(*SENSOR_DATA_SIZE),has_coll,engine.getGoalDist() <=0,engine.getGoalDist(),engine.getGoalDistFromStart())
        reward=brain.analize_stateV6(currState)
        tmp,disp=brain.act(currState,False) #predict best move
        override_counter+=1#incrament counter
        tmp=tmp.squeeze()
        #display depth buffer
        if(startMover(engine)): 
            #engine.addObstacle(obs,showCollider=True)
            plt.imshow(currState.sensorDat,cmap="Greys")
            plt.show()

        #input for terminating training
        if(earlystop(engine)):
            pause=hideBar(bar)
            next(pause)
            confirm=input("stop training?[y/n]:")
            if confirm.lower() == "y" or confirm.lower() == "yes":
                stop=True
                dobreak=True
            
            
        #preform movement
        engine.drone.move(np.clip(tmp[:3],-0.1,0.1))
        engine.drone.rotate(np.clip(tmp[3:],-0.5,0.5))
        engine.drone.rotation=np.clip(engine.drone.rotation,-50,50)
        engine.drone.move(np.array([0,0.05,0]))

        #display info
        pred_disp=torch.round(disp.detach(),decimals=2).tolist()[0]
        pred_disp=[round(elm,4) for elm in pred_disp]
        bar.update(bar_task,advance=0,curr_reward=round(reward,4),last_pred=pred_disp)
        timer-=1
        #time.sleep(0.016666)#limit update rate to 60 fps
    #save stuff
    brain.save(RUN_IDETIFIER+r"/model",epoch)
    f=open(RUN_IDETIFIER+r"/module_list","w")
    dat=json.dumps(modules)
    f.write(dat)
    f.close()
    save_rewards(RUN_IDETIFIER+r"/rewards.json")

    losses.append(brain.learn())#backprop

    engine.unloadScene()#cleanup

    return generate_new,dobreak

epochs=50

#fancy progress bar 
best_reward=2
levels_beaten=0
tries=0
avg_tries=0
best_score_display=TextColumn("Best Reward: {task.fields[best_reward]}")
current_score_display=TextColumn("Current Reward: {task.fields[curr_reward]}")
levels_beaten_display=TextColumn("levels beaten: {task.fields[levels_beaten]}")
avg_tries_display=TextColumn("time since last win: {task.fields[avg_tries]}")
last_pred_display=TextColumn("last prediction: {task.fields[last_pred]}")
was_terminted=""
#hides the progress bar
def hideBar(progress: Progress):
    transient = progress.live.transient # save the old value
    progress.live.transient = True
    progress.stop()
    progress.live.transient = transient # restore the old value
    try:
        yield
    finally:
        # make space for the progress to use so it doesn't overwrite any previous lines
        print("\n" * (len(progress.tasks) - 2))
        progress.start()

bar=Progress(TextColumn("[progress.description]{task.description}"),BarColumn(),MofNCompleteColumn(),current_score_display,levels_beaten_display,avg_tries_display,last_pred_display)
bar_task=bar.add_task("training...",total=epochs,curr_reward="N/A",levels_beaten=levels_beaten,avg_tries=avg_tries,last_pred="N/A")
bar.start()
new_counter=3
time_since_last=0
#document levels generated
generated_levels=[]
generated_levels.append(modules)

#training loop
for epoch in range(epochs-start_epoch):
    generate_new,dobreak=gameLoop(engine,brain,epochs-epoch)#run training episode
    #update info display
    best_reward=min(best_reward,min(brain.lifetime_rewards[-1]))
    bar.update(bar_task,advance=1,curr_reward=brain.lifetime_rewards[-1],levels_beaten=levels_beaten,avg_tries=time_since_last)
    tries+=1
    if(not generate_new):
        time_since_last+=1
    if(levels_beaten >0):
        avg_tries=tries/levels_beaten#calculate new avarege tries
    if(generate_new):
        new_counter-=1
        best_reward=0
        levels_beaten+=1
        time_since_last=0
        
    #generate new scene if network has succeeded.
    if (generate_new ):
        generated_levels.append(modules)
        scene1,modules=compile_scene(["L_top_right.txt"],"level_modules")
        
    #terminate traning if needed
    if dobreak:
        was_terminted=f"\ntraining was terminated at epoch {epoch} / {epochs}."
        break

#generate report
run_data=f"""
best score: {best_reward}\n
explore factor: {brain.explore_factor}\n
levels beaten: {levels_beaten}\n
avg tries: {avg_tries}
"""+was_terminted+"generated levels:\n"
for lvl in generated_levels:
    run_data+=f"level content: {lvl}\n"



f=open(RUN_IDETIFIER+r"/run_data.txt","a")
f.write(run_data)
f.close()

#plot rewards graph
rewards=[]
for x in brain.lifetime_rewards:
    rewards+=x
plt.plot(rewards)
plt.savefig(RUN_IDETIFIER+"/reward_graph.svg",format='svg')
plt.show()

#plot loss graph
plt.plot(losses)
plt.savefig(RUN_IDETIFIER+"/loss_graph.svg",format='svg')
plt.show()


#plot avarege reward per epoch
avgs=[]
for x in brain.lifetime_rewards:
    avgs.append(sum(x))
plt.plot(avgs)
plt.savefig(RUN_IDETIFIER+"/avg_reward_graph.svg",format='svg')
plt.show()

#save model
brain.save(RUN_IDETIFIER+r"/model",epochs)

engine.destroy() #cleanup



