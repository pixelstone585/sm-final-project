#old code, for development use only
def learn(self):
        if (self.curr_step % self.sync_interval == 0):
            self.sync_target()

        if (self.curr_step % self.save_interval ==0):
            self.save()

        if(self.curr_step < self.burn):
            return None,None

        if(self.curr_step % self.learn_interval != 0):
            return None, None

        #recall
        state,next_state,action,reward,done = self.recall()

        #estimate q
        td_est=self.td_estimate(state,action)

        #get target q
        td_tgt=self.td_target(reward,next_state,done)

        #backprop
        loss=self.update_online(td_est,td_tgt)

        return (td_est.mean().item(),loss)
def recall(self):
        #retrive expiriance from memory
        batch=self.memory.sample(self.recall_batchSize).to(self.device)

        state,next_state,action,reward,is_done = (batch.get(key) for key in ("state","next_state","action","reward","is_done"))# execute batch.get(key) for each value is the list
        return state, next_state, action.squeeze(), reward.squeeze(), is_done.squeeze()
