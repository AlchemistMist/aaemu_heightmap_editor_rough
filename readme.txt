Archeage Heightmap Editor Rough 
Made by Alchemist 
v0.1

Overall
A collection of scripts that pulls heightmap data, turns them into .tif images, combines the images into cells to edit with cryengine or other tools. And some scripts to put that data back in. 
It pulls the sector vertex data, turns it into 16bit greyscale tiff images, removes the edge vertices and merges them to make cells, which can be merged into groups of cell, then changed to a 16bit greyscale .raw file that can be loading into cryengine to edit.
This tool makes height maps with a max height of 2048. Set the max height in cryengine to that before importing, you can edit max height in 01cellpuller.py
It builds the height map with heights of vertices and removes the bottom and right rows. (Maybe not the best way.) 

After editing in cryengine and exporting as a 16bit .raw file. The scripts will turn that file into .tif images, break them into cells then into sectors, add an extra each pixel for the vertices, and make a new heightmap.dat file to paste into your game files.
When inserting edited data, it will use the original resolution for each sector. So low res sectors will keep the original resolution. So modding some cells, especially ones on edges of the map or out of bounds, will need more things edited in the .dat file. It also changes the min height and height scaling for the vertices. There’s possible other data that needs to be changed too.


This is only a heightmap editor. You can’t change the terrain texture or objects placement. That’s a future project. 
You don’t need cryengine for this, you can edit the heightmap data with other methods.


Disclaimer
This is a very rough tool, more of a proof of concept to be worked on later when I understand the files a bit more. With the help of ai i put these scripts together to change the terrain in-game. I’m a beginner with modding and coding, so there’s some obvious things that’ll be wrong. I’ll rework and make a better tool as the process and workflow becomes more clear.
Also this process isn’t fully worked out. In the heightmap.dat file there’s a bit of data after the sector vertex data about 16 bytes which might be important. This tool just leaves the old data.


Requirements
python, pillow, numpy, opencv-python, tifffile. (maybe something else)



Set up
Put scripts in game/main_world/cells. Or make a new folder and move the cells folders map you want to change.
Make a backup of the main_world folder in case something gets broken.


Pulling heightmap data (short)
Run scripts 01, 02, 03, 04, 06. SKIP 05 
Only use script 05 if you are going to use the archeage editor.
You may need to manually move files and make new folders if I didn't set everything up right. 


Also images will appear rotated, you DON’T need to change that. Just be aware north is right, west is up, ect.


Pulling heightmap data (longer)
Script go in the cells folder, so you should have scripts and a big list of folders like 000_000, 000_001, ect

Hold shift and right click the background of the folder (with nothing selected)
click "Open Powershell Window here"

Run 01cellpuller.py by typing "py 01cellpuller.py"
That will pull data for each sector, save them as images in output_sectors folder. Sectors are subdivisions of a cell, 16x16 per cell
It'll take about 10min if you try to processing the whole main_world

Run 02sectortrimmer by typing "py 02sectortrimmer"
This will crop the images, it's using the vertex placement so it needs to be corrected for that.
Basically each sector has an overlapping pixel of the next sector, this script removes that.
This also takes about 10min if you do the whole map.

Run 03sectormerge.py
This merges the images so the cell is complete.

Don’t run 04cellmerge.py yet
You need to edit this file, based on the cells you want to edit. This makes a heightmap for more than one cell. It’s good to edit more than one so you can see the surrounding terrain. 
For example, if you set START_CELL = 000_000, END_CELL = 003_003. that will make a 3x3 image. Start_cell is bottom left and end_cell is top right. 
It's set up to only make squares. I use the AAEmu db editor map viewer to know what cells to start and end with.
Once set and saved, run the script. You can check the output in “raw_input” to see the image, again the image is rotated, so it’ll look confusing to look at.

Only run 05tifff4092.py if you are using the archeage editor. 
You need to make a folder called "tiff_resize", remove and place the mergecell image in the folder from “raw_input”. This script outputs to “raw_input too, so it’ll replace images in there with the same name. This script is for the archeage 0.5 editor, a custom version of cryengine 2 used for development for archeage 0.5, it seems to need 4092x4092 images regardless of the size of level. 
This increases the size and the image to the right spot for the editor... 

Run 06tifftoraw.py
This changes tiff to .raw file for cryengine to use


Editing in cryengine
I’ve been using cryengine 3.5.8. you could use newer versions too i think or you can use other height map editing tools or game engines.



Basic steps
Make new level/world.
Set “Heightmap Resolution” to the resolution of your image or;
512x512 = 1 cell, 1024x1024 = 2x2 cells, 2048x2048 = 4x4 cells, 4096x4096 = 8x8 cells
Meters per unit = 2
Click ok, then set terrain texture to 4094x4096 (but i don’t think this matters)

It should load a blank level with water.
At the top click “Terrain” then click “Edit Terrain”
This opens a window with a sorta mini map. Click “Modify” at the top of the window.  
Set max height to 2048 (you need to import heightmap again if you imported it before setting this)
Set water height to 100
Then you might need to “Reload terrain” that’s in the “terrain” menu at the top (the one you used to open the terrain editor window)
Edit terrain with the cryengine tools, there’s old guides online.



Importing edited data into map files.
Export 16bit raw height map file from cryengine after editing.
Save to a new folder “edited_raw”. The file name of start and end cell. It’s the same name as the raw put imported into cryengine. Then run scripts 10-15 but skip 11

10editedrawtotiff.py
Makes edited raw files into tiff

SKIP 11editedtiffresize.py
This was for the archeage editor, it’s there, but i haven’t used it.

12editedcellmaker.py
Makes cells images and names them with the correct cell name. It makes 512x512, 512x513, 513x512 and 513x513.
513x513  is for 1 pixel overlap for vertices. It marks folders with “f” for the next script to fix.

13editedsectormaker.py
Makes cells into sector, makes 33x33 for vertices, marks 32x32, 33x32, 32x32 sectors with 'f' than need to be fixed

14editedsectorfixer.py
This fixes sectors so they are 33x33, the right size for heightmap.dat
It uses "output_sectors" to get 1 pixel extra data from old data (that data shouldn't change while editing since it's at the edge of the outer cells)
use 01cellpuller.py to get data if it isn’t working.

15datainsertv4.py
Puts data into heightmap.dat in edited_height folder. It set up subfolders to make copy/paste easier.
This still needs work to adjust a few things. 

Copy new heightmap.dat files and paste to replace the ones in game files.
You can just copy the folders and paste them into the cells folder and replace all.

Done 


You can also edit archeages game world files while in character select. You don’t need to reopen the game. Also having the client using an unpacked game.pak is imported to make modding easier. 




