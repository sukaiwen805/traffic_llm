# 路网生成方法

### 1、生成路网net.xml文件
- 运用netedit手动生成路网
   复杂交通情况不建议
   建议使用打开 [openstreetmap](https://www.openstreetmap.org/) 官网，搜索地名，手动框选需要导出道路区域并导出，下载生成的 *.osm 文件

- 导出的 .osm 文件中除了路网信息，还包含建筑、河流等信息，需要用 polyconvert 工具进行处理，方法如下：
- 打开sumo安装路径中 ..\Sumo\doc\userdoc\Networks\Import 中OpenStreetMap文件,复制文件中 Importing additional Polygons (Buildings, Water, etc.) 部分代码到记事本 ，另存为 typemap.xml ，保存到 .osm 文件相同路径。
- 打开…/bin/start-command-line.bat ，使用命令进入 *.osm 文件的文件夹，输入指令
  ```bash
   netconvert --osm-files map.osm -o map.net.xml
   ```
- 继续输入
  ```bash
   polyconvert --net-file map.net.xml --osm-files map.osm --type-file typemap.xml -o map.poly.xml
   ```
   此时文件夹中已生成路网文件和地形文件

### 2、生成车流文件 rou.xml
- 输入命令
   ```bash
   python ...\sumo-0.30.0\tools\randomTrips.py -n map.net.xml -e 100 -l
   ```
  生成随机行驶的车辆，-e 100 -l 为随机工具的配置，100为车辆数，可按实际情况设置
- 最后，使用 bin 文件夹下的 duarouter.exe 把随机的旅程和道路信息结合起来获得了车流文件（rou.xml）
- 输入命令
  ```bash
   python ...\sumo-0.30.0\tools\randomTrips.py -n map.net.xml -r map.rou.xml -e 100 -l
   ```
- 在文件夹中可以看到车流文件 map.rou.xml
- 最后编写配置文件 (*.sumo.cfg) 在 记事本 中输入下面的代码，保存为 map.sumo.cfg 文件
- 打开 sumo-gui.exe ，点击 File->Open Simulation Configuration ，找到配置文件，点击 “OK”

### 建议：快速生成来自真实世界数据的路网，使用osm2netconvert与netconvert工具。对于信号灯、车辆流通方向等非常影响实验结果的配置，得到net.xml文件后，在NetEdit中设计
