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

