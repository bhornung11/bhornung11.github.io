---
layout: post
title:  "Musings of data quality analysis"
date:   2025-11-08 10:01:21 +0000
categories: data quality analysis
---
This short post discusses various aspects of data quality analysis. Primary role of data quality analysis is to help choose optimal actions on the data - as a part of a product - through quantifying its goodness. As such, it should be cost efficient both during development and in its application. This is to be achieved by building a system that is robust, complete, focuses on the right features of the data and considers its environment and the resources.

## Introduction

Upon examining the US _fiat_ currency it is readily noticed that trust is placed in God not in data. Indeed, verily say I to you one shall not and cannot trust data. One can, however, write down commandments and measure how well the set at hand adheres to them. The degree to which it does so shall be named the data goodness. It should quantify how usable the collection of records either through the value added or removed during the development and the application of the final product.

## Basic considerations

Based on the immense wisdom of the scribbler of these lines, a few tenets of data quality analysis practices are enumerated here. 

1. It is just as tearful as true, money rules.

1. Data quality analysis must serve a purpose
   
1. No dataset is known until each and every of its element is seen.
    1. Whatever can happen to the data it will eventually do. It is just a question of time.
    
1. Do not rely on assumptions. Rather, check for them.
    1. Expect the unknown and contain it as opposed to preparing for hypothetical scenarios.

1. Define quality criteria in reponse to needs. Checking against them should provide value.
   
1. Quantify the goodness of the data.

1. The data quality analysis does not concern itself with making decisions.
    1. It generates information which helps choose the right course of actions.
    1. It provides insight in the data which in itself may be valuable.

1. Quantify the cost of the various types and degrees of data deficiency.
    1. What consequences they have to the application consuming the data?
    1. What actions are required to handle these issues?
    1. What is the cost of these actions?
       
1. Data quality analysis is a component of system from sourcing the data until having used the utmost application with satisfaction.
       
1. Likewise, forget not about how outcome of the analysis is processed.

1. Data quality analysis has a cost.
   1. Try to make it cheap as possible.

1. The data needs to be moved from a place to an other where the analysis takes place.
    1. This incurs a fee. It is paid in time or memory. Eventually settled using a bank account.
    1. Be mindful of the data source, how the data is accessed.

1. Make the logic as disjoint as possible. By doing so, the framework becomes more resilient to logical errors; and the results will be clearer.
   1. It must be possible to execute a check regardless of the others, unless explicitly instructed not to do so.
   1. The outcome of a check should not rely on previous ones.
   1. The validity of a check should not rely on those of other ones.
   1. The interpretation of a check should not rely on those of other ones.

1. Keep the analysis as simple as possible.
   
1. Answer questions as early as possible.
   
1. Enjoy it!
