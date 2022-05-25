# building an html book for github.io

## Setup

1. Fork https://github.com/UBC-DSCI/introduction-to-datascience-python  
2. Upload a ssh public key to your github account  
3. Edit your .ssh/config to allow ssh read/write to github with your user id:  

     Host phaustin
         HostName github.com
         User git
         IdentityFile ~/.ssh/new_pha_git
         IdentitiesOnly yes
         
4. Clone your fork using ssh:

      git clone phaustin:phaustin/introdutin-to-datascience-python

5.  Add the upstream remote:

      git remote add upstream phaustin:UBC-DSCI/introduction-to-datascience-python
      
6.  Fetch the upstream branches

      git fetch upstream
      
6.  Create the topic branch:

      git checkout -b classification1 origin/classification1
      
7.  If necessary (i.e. if someone has pushed new changes to main), rebase origin on upstream

       git rebase upstream/classification1

8.   Build the book and push the html to gh-pages

       jb build source
       ./push_html.sh
       
9.   In the setting for your forked repo, turn on pages and set it to source gh-pages

10.  Point your browser to https://phaustin.github.io/introduction-to-datascience-python/classification1.html#


(note that it might take a minute or two for github to overwrite the old html on their server.  There can also be browser cache issues.  I use the "clear cache" extension on chrome and also do a hard refresh: https://fabricdigital.co.nz/blog/how-to-hard-refresh-your-browser-and-clear-cache  I also sometimes write version number in the first header so I know that I'm looking at the current version.  You can also use an incognito window to be sure you've got a fresh session.)
