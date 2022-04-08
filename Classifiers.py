# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:22:31 2020

@author: Hang Yu
"""
import scipy.stats
import numpy as np
import random
import math
from scipy.stats import wasserstein_distance

class Classifiers:
    def __init__(self):
        #self.action_num=action_num
        self.pos_dist=[]
        self.neg_dist=[]
        self.grade = []
        self.window_size = 200
        np.random.seed()
    def confidence(self,fb,  pos, neg, case):
        print
        if case == 1:
            return 1 + pos - neg
        if case == 2:
            return -1 - neg + pos
        if case == 3:
            return  pos
        if case == 4:
            return  -1 * neg     

      
    def steady(self,fb):
        sigma=1
        #print(fb)
        if len(self.pos_dist)+len(self.neg_dist) < 100:      
            # if len(self.pos_dist)+len(self.neg_dist) != 0:
            #     mu= np.percentile(self.pos_dist + self.neg_dist, 50)
            #     #print(mu,fb)
            # else:
            mu = 5
            if fb > mu:
                self.pos_dist.append(fb)
                return 1
            if fb <= mu: 
                self.neg_dist.append(fb)
                return -1
        else:
            if len(self.neg_dist)==0:
                min_pos = min(self.pos_dist)
                self.pos_dist.remove(self.pos_dist[self.pos_dist.index(min(self.pos_dist))])
                self.neg_dist.append(min_pos)
            if len(self.pos_dist) == 0:
                max_neg=max(self.neg_dist)
                self.neg_dist.remove(self.neg_dist[self.neg_dist.index(max(self.neg_dist))])
                self.pos_dist.append(max_neg)
            
        # if len(self.neg_dist) >= 200:
        #     self.neg_dist.remove(self.neg_dist[0])
        # if len(self.pos_dist) >= 200:
        #     self.pos_dist.remove(self.pos_dist[0])
        mu_pos=np.mean(self.pos_dist)
        std_pos=np.std(self.pos_dist,ddof=1)
        if math.isnan(std_pos) or std_pos==0:
            std_pos = sigma
        mu_neg=np.mean(self.neg_dist)
        std_neg=np.std(self.neg_dist,ddof=1)
        if math.isnan(std_neg) or std_neg==0:
            std_neg = sigma
        
        prob_pos_ = scipy.stats.norm(mu_pos, std_pos).pdf(fb)
        prob_neg_ = scipy.stats.norm(mu_neg, std_neg).pdf(fb)
        #print(fb, prob_pos, prob_neg)
        prob_pos = prob_pos_ / ( prob_pos_ +  prob_neg_)
        prob_neg = prob_neg_ / ( prob_pos_ +  prob_neg_)
        #print(fb, prob_pos, prob_neg)
        p = np.random.random()
        dis_flag = 0
        # if p < prob_pos:
        #     self.pos_dist.append(fb)
        #     dis_flag = 1
        #     #self.confidence(fb, prob_pos,prob_neg,1)
        # else:
        #     self.neg_dist.append(fb)
        #     dis_flag = -1
        if abs(fb - mu_pos) < abs(fb- mu_neg):
            self.pos_dist.append(fb)
            dis_flag = 1
            #self.confidence(fb, prob_pos,prob_neg,1)
        else:
            self.neg_dist.append(fb)
            dis_flag = -1
        if fb <=mu_neg +3*std_neg and fb >= mu_pos - 3*std_pos:   
            if dis_flag == 1:
                if min(self.pos_dist) != fb:
                    min_pos = min(self.pos_dist)
                    self.pos_dist.remove(self.pos_dist[self.pos_dist.index(min(self.pos_dist))])
                    self.neg_dist.append(min_pos)
               # print("case 3: ")
                return  self.confidence(fb, prob_pos,prob_neg , 3)  
            else:
                if max(self.neg_dist) != fb: 
                    max_neg=max(self.neg_dist)
                    self.neg_dist.remove(self.neg_dist[self.neg_dist.index(max(self.neg_dist))])
                    self.pos_dist.append(max_neg)
               # print("case 4: ")
                return  self.confidence(fb, prob_pos,prob_neg , 4)
        
        if fb < mu_pos and fb > mu_neg:
            if dis_flag == 1:
                #print("case 3: ")
                return self.confidence(fb, prob_pos,prob_neg , 3)  
            else:
                #print("case 4: ")
                return self.confidence(fb, prob_pos,prob_neg , 4) 
            
        if dis_flag == 1:
            #print("case 1: ")
            return self.confidence(fb, prob_pos,prob_neg , 1)  
        else:
            #print("case 2: ")
            return self.confidence(fb, prob_pos,prob_neg , 2) 
            
                
                
        

    def s_w(self,fb):
        self.grade.append(fb)
        if len(self.grade) > self.window_size:
            self.grade.remove(self.grade[0])
        mu= np.percentile(self.grade, (50), interpolation='midpoint')
        if fb >= mu:
            return 1
        if fb < mu:  
            return -1
        
    def naive(self,fb):
        if fb >= 5:
            return 1
        if fb < 5:
            return -1
        
    def STEADY(self, feedbacks):
        fbs = []
        for f in feedbacks:
            fbs.append( self.steady(f))
        return fbs
    def WINDOW (self, feedbacks):
        
        fbs = []
        for f in feedbacks:
            fbs.append( self.s_w(f))
        return fbs 
    def MIDPOINT(self, feedbacks):
        fbs = []
        for f in feedbacks:
            fbs.append( self.naive(f))
        return fbs
        




    # def wass(self,fb):

    #     if len(self.neg_dist) >= 1000:
    #         self.neg_dist.remove(self.neg_dist[0])
    #     if len(self.pos_dist) >= 1000:
    #         self.pos_dist.remove(self.pos_dist[0])
    #     if len(self.pos_dist)+len(self.neg_dist) < 100:      
    #     #     if len(self.pos_dist)+len(self.neg_dist) != 0:
    #     #         mu= np.percentile(self.pos_dist + self.neg_dist, (50), interpolation='midpoint')
    #     #         #print(mu,fb)
    #     #     else:
    #     #         mu = 5
    #         mu = 5
    #         #print(np.mean(self.neg_dist),np.mean(self.pos_dist), fb)
    #         if fb >= mu:
    #         # if cnt > 0.6*(len(self.pos_dist)+len(self.neg_dist)):
    #             self.pos_dist.append(fb)
    #             return 1
    #         if fb <= mu: 
    #         # if cnt <= 0.6*(len(self.pos_dist)+len(self.neg_dist)):
    #             self.neg_dist.append(fb)
    #             return -1
    #     else:
    #         if len(self.neg_dist)==0:
    #             min_pos = min(self.pos_dist)
    #             self.pos_dist.remove(self.pos_dist[self.pos_dist.index(min(self.pos_dist))])
    #             self.neg_dist.append(min_pos)
    #         if len(self.pos_dist) == 0:
    #             max_neg=max(self.neg_dist)
    #             self.neg_dist.remove(self.neg_dist[self.neg_dist.index(max(self.neg_dist))])
    #             self.pos_dist.append(max_neg)
            
    #     # if len(self.neg_dist) >= 200:
    #     #     self.neg_dist.remove(self.neg_dist[0])
    #     # if len(self.pos_dist) >= 200:
    #     #     self.pos_dist.remove(self.pos_dist[0])
    #     self.pos_dist.append(fb)
    #     dis_p = wasserstein_distance(self.pos_dist,self.neg_dist)
    #     self.pos_dist.pop()
        
    #     self.neg_dist.append(fb)
    #     dis_n = wasserstein_distance(self.pos_dist,self.neg_dist)
    #     self.neg_dist.pop()
        
        
    #     if dis_p > dis_n:
    #         mu = np.mean(self.pos_dist)
    #         self.pos_dist.append(fb)
    #         if fb > mu:
    #             return self.confidence(fb, mu, 10, 1)
    #         else:
    #             return self.confidence(fb, mu, 10, 11)
    #     else:
    #         mu = np.mean(self.neg_dist)
    #         self.neg_dist.append(fb)
    #         if fb < mu:
    #             return self.confidence(fb, mu, 10, -1)
    #         else:
    #             return self.confidence(fb, mu, 10, -11)           
                
        
    #     return 0  


