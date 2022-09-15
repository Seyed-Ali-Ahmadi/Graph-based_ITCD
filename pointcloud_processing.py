# point cloud processing tools
import pdal


# TODO Ground Filtering: Import raw point cloud file and classify ground points
#     "<input raw point cloud>.las",
#     "<output ground points file>.las"
json1 = """
{
  "pipeline":[
    "/PointCloud.las",
    {
      "type":"filters.smrf",
      "scalar":1.2,
      "slope":0.2,
      "threshold":0.45,
      "window":16.0
    },
    "/Ground.las"
  ]
}"""
pipeline = pdal.Pipeline(json1)
pipeline.validate()
pipeline.execute()


# TODO Rasterization: After obtaining ground points, calculate nDSM using LAStools.
#  Then make a raster from it.
# Calculate nDSM using LAStools. I don't remember if PDAL can do this or not.
# It is possible to obtain both DTM and DSM and then subtract them from each other.
# "filename": "<output raster name directory>.tif",
json2 = """
{
    "pipeline": [
        "/nDSM.laz",
        {
            "type": "writers.gdal",
            "filename": "/CHM.tif",
            "dimension": "Z",
            "radius": 0.15,
            "data_type": "float",
            "output_type": "max",
            "resolution": 0.15
        }
    ]
}
"""
pipeline = pdal.Pipeline(json2)
pipeline.validate()
pipeline.execute()
