# TDSortParticles

This is a cPlusPlus plugin for TouchDesigner. Its purpose is to sort particles in TOP context with regard to their distance to a given reference object.
When rendering particles with transparancy, this is crucial for correct alpha blending.

Have a look at sortParticles.toe to see how to use the plug-in. You should be able to copy the sortParticles.dll file in your plugin folder to use it as a custom TOP. 

Vincent Houzé published a very similiar plugin here: https://github.com/vinz9/CudaSortTOP
My new implementation support the newest TouchDesigner versions 2022 and has a more intuitive workflow (I think)

Please tag me if you share any project using this online. 
Reach out to me if you have any questions: josefpelz.com/contact

![sortParticles_example](https://user-images.githubusercontent.com/28048180/203425104-a6ecffbe-5cbe-4184-ae8b-001f682f6802.png)
