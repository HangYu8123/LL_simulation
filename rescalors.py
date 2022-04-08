# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:07:09 2021

@author: Hang Yu
"""

import numpy as np
import random
import math, time
import torch

def rescale(truth, guess):
    oracle = truth.detach().numpy().tolist()[0]
    agent =  guess.detach().numpy().tolist()[0]
    
    cnt1 = 0
    cnt2 = 0
    if oracle[0] < 0 and agent[0]<0:
        pass
    else:
        # if oracle[0] < 0:
        #     cnt1 += abs(agent[0])
        # else:
        cnt1 += abs(agent[0] - oracle[0])
    if abs(oracle[1]) < 0.5  and abs(agent[0]) < 0.5:
        pass
    else:
        # if abs(oracle[1]) < 0.5 :
        #     cnt2 +=  abs(abs(agent[1]) - 0.5)
        # else:
        cnt2 += abs(agent[1] - oracle[1])
    # cnt1 += abs(agent[0] - oracle[0])
    # cnt2 +=  abs(agent[1] - oracle[1])
    #print(oracle, agent, cnt1, cnt2)
    return round(10 - (cnt1 + cnt2)/4 * 10)
    #return (10 - max(cnt1 , cnt2)/2 * 10) 
    
def rescale_binary(truth, guess):
    oracle = truth.detach().numpy().tolist()[0]
    agent =  guess.detach().numpy().tolist()[0]
    

    if oracle[0] < 0 and agent[0] > 0:
        return -1
    if oracle[0] > 0 and agent[0] < 0:
        return -1
    if oracle[1] > 0.5  and agent[0] < 0.5:
        return -1
    if abs(oracle[1]) < 0.5 and abs(agent[0]) > 0.5:
        return -1
    if oracle[1] <- 0.5  and agent[0] > -0.5:
        return -1

            
    return 1
    #return (10 - max(cnt1 , cnt2)/2 * 10) 
def rescale_rwd(rwd):
    if abs(rwd) < 2:
        return min(max((rwd+2)/4 * 10, 0),10)
    else:
        return 0
def dynamic_feedback_binary(feedback, focus):
    p =random.random()
    if p < focus:
        return feedback   
    else:
        return  np.random.choice([-1,1], p = [0.5,0.5])
def dynamic_feedback(feedback, focus, bias, consistency):
    p =random.random()
    if p < focus:
        feedback  = feedback + np.random.normal(bias, consistency)     
    else:
        feedback =  np.random.random_integers(0,10)
    return round(feedback)