# building an html book for github.io

## Setup

1. Fork https://github.com/UBC-DSCI/introduction-to-datascience-python  
2. Upload a ssh public key to your github account  
3. Edit your .ssh/config to allow ssh read/write to github with your user id -- using a nickname like phaustin for your github  host entry:  

```bash
      Host phaustin
         HostName github.com
         User git
         IdentityFile ~/.ssh/new_pha_git
         IdentitiesOnly yes
```
         
4. Clone your fork using ssh:

```bash
      git clone phaustin:phaustin/introduction-to-datascience-python
```

5.  Add the upstream remote:

```bash
      git remote add upstream phaustin:UBC-DSCI/introduction-to-datascience-python
```
      
6.  Fetch the upstream branches

```bash
      git fetch upstream
```

6.  Create the topic branch:

```bash
      git checkout -b classification1 origin/classification1
```

7.  If necessary (i.e. if someone has pushed new changes to main), rebase origin on upstream

```bash
       git rebase upstream/classification1
```

8.   Build the book and push the html to gh-pages

```bash
       jb build source
       ./push_html.sh
```

9.   In the setting for your forked repo, turn on github pages and set it to source the gh-pages branch

10.  Point your browser to the fork's github.io address:

 https://phaustin.github.io/introduction-to-datascience-python/classification1.html#


** note that it might take a minute or two for github to overwrite the old html on their server.  There can also be browser cache issues.  I use the "clear cache" extension on chrome and also do a hard refresh: see https://fabricdigital.co.nz/blog/how-to-hard-refresh-your-browser-and-clear-cache.  I also sometimes write a version number in the top header so I know that I'm looking at the current version.  You can also use an incognito window to be sure you've got a fresh session.
