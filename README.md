# Social Media Analytics

### [Specification](https://github.com/LOOP115/Social_Media_Analytics/blob/main/resources/ass1-spec.pdf)



### Requirements

`python 3.7.4`

`mpi4py 3.0.2`

`numpy 1.21.6`

`pandas 1.3.5`

**Data files should be in the `data` folder which is in the same directory with `main.py`.**



### Run locally

```
mpiexec -n [NUM_PROCESSES] python main.py [DATA]

mpiexec -n 1 python main.py tiny

mpiexec -n 8 python main.py small

mpiexec -n 8 python main.py big
```

**If no data file is specified in arguments, `tinyTwitter.json` will be selected as default.**



### Run on Spartan

| Resource                                    | Slurm                          | Output                             |
| ------------------------------------------- | ------------------------------ | ---------------------------------- |
| 1 node and 1 core                           | [1n1c.slurm](slurm/1n1c.slurm) | [1n1c.out](outputs/batch/1n1c.out) |
| 1 node and 8 cores                          | [1n8c.slurm](1n8c.slurm)       | [1n8c.out](outputs/batch/1n8c.out) |
| 2 nodes and 8 cores (with 4 cores per node) | [2n8c.slurm](2n8c.slurm)       | [2n8c.out](outputs/batch/2n8c.out) |

```
sbatch 1n1c.slurm

sbatch 1n8c.slurm

sbatch 2n8c.slurm
```



### Problem Description

Your task in this programming assignment is to implement a parallelized application leveraging the University of Melbourne HPC facility SPARTAN.

Your application will use a large Twitter dataset and a file containing the suburbs, locations and Greater Capital cities of Australia. Your objective is to:

* count the number of different tweets made in the Greater Capital cities of Australia,
* identify the Twitter accounts (users) that have made the most tweets, and
* identify the users that have tweeted from the most different Greater Capital cities.

#### Data

* Some suburb names (strings) are repeated many times. Your solution should aim to tackle such issues and dealing with potential ambiguities and/or flag cases where tweets cannot be resolved to a single unique location.
* The tweets themselves can include the precise location or the approximate location of where the tweet was made. Such approximate and non-Australian locations can be ignored from the analysis, i.e., unless the specific string is in the sal.json file it can be ignored.

#### Target

* Your assignment is to search the large Twitter data set (bigTwitter.json) and using sal.json file, count the number of tweets in the various capital cities.
  * You may ignore tweets made in rural locations, e.g., 1rnsw (Rural New South Wales), 1rvic (Rural Victoria) etc.
* Each tweet also contains a unique author id for the tweeter. Your solution should count the number of tweets made by the same individual and return the top 10 tweeters (in terms of the number of tweets made).
* Finally, your solution should identify those tweeters that have tweeted in the most Greater Capital cities and the number of times they have tweeted from those locations.
  * The top 10 tweeters making tweets from the most different locations should be returned and if there are equal number of locations.
  * Note that only those tweets made in Greater Capital cities should be counted, e.g., if author Id = 5678910111213141516
    tweets 1000 times from rural New South Wales then these can be ignored.

* Your application should allow a given number of nodes and cores to be utilized. Specifically, your application should be run once to search the bigTwitter.json file on each of the following resources:
  * 1 node and 1 core
  * 1 node and 8 cores
  * 2 nodes and 8 cores (with 4 cores per node)

#### Spartan

* The resources should be set when submitting the search application with the appropriate SLURM options.
  * Note that you should run a single SLURM job three separate times on each of the resources given here
* You can implement your solution using any routines and libraries you wish however it is strongly recommended that you follow the guidelines provided on access and use of the SPARTAN cluster.
  * You may wish to use the pre-existing MPI libraries that have been installed for C, C++ or Python, e.g., mpi4py.
  
  * You should feel free to make use of the Internet to identify which JSON processing libraries you might use.
  * You may also use any regular expression libraries that you might need for string comparison.
* Your application should return the final results and the time to run the job itself, i.e. the time for the first job starting on a given SPARTAN node to the time the last job completes.
  * You may ignore the queuing time. The focus of this assignment is not to optimize the application to run faster, but to learn about HPC and how basic benchmarking of applications on a HPC facility can be achieved and the lessons learned in doing this on a shared resource.

#### Report

You should write a brief report on the application – no more than 4 pages!, outlining how it can be invoked, i.e. it should include

* the scripts used for submitting the job to SPARTAN,
* the approach you took to parallelize your code,
* and describe variations in its performance on different numbers of nodes and cores.

Your report should include the actual results tables as outlined above and a single graph (e.g., a bar chart) showing the time for execution of your solution on 1 node with 1 core, on 1 node with 8 cores and on 2 nodes with 8 cores.



### Submission

The assignment should be submitted to Canvas as a ***zip*** file. The zip file must be named with the students named in each team and their student Ids. That is, *ForenameSurname-StudentId:ForenameSurname-StudentId* might be *SteveJobs-12345:BillGates-23456.zip*. Only one report is required per student pair and *only one student needs to upload this report*.

