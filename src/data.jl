"""
    function loadblpdata()
  
Loads data from Berry, Levinsohn, and Pakes (1999).

Returns a DataFrame.
"""
function loadblpdata()
  csvfile=normpath(joinpath(dirname(Base.pathof(BLPDemand)),"..","data","blp_1999_data.csv"))
  dt = CSV.read(csvfile)
end

