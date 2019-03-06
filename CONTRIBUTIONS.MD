# How to contribute?

## 1) Download the project.
`git clone ....`
`cd CarND-Capstone`

## 2) Create a new branch
[WARNING] 

Please DON'T code on master branch, it's super important so that we do not have code conflicts.

`git checkout -b feature/my_feature_branch`

Every feature must be in a separated branch, example:
- feature/waypoint-updater
- feature/traffic-light-detector
- feature/drive-by-wire

## Start coding

Ok, now you can code your feature, commit all your changes and etc, but here has a interesting git flow to follow to we don't have code conflicst between `git merges`.

[WARNING]

Every single day that you will work on udacity project, remeber to follow this steps:
- Update your branch with master
    - `git fetch origin`
    - `git merge origin/master`

## Finish your eature
When you finish your feature, first of all update your current branch with master(follow the **Start coding** step), so merge it on master branch:
`git checkout master`
`git merge feature/my-feature`

## Need some help?
Talk with @italojs in SDC slack or send a e-mail to italo.i@live.com