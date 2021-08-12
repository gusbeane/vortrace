
#include "pointcloud.hpp"
#include <boost/multi_array.hpp>
#undef H5_USE_BOOST
#define H5_USE_BOOST
#include <highfive/H5File.hpp>
#ifdef TIMING_INFO
#include <chrono>
#endif

//Load gas from snapshot, applying subbox {xmin,xmax,ymin,ymax,zmin,zmax}
//Build tree
void PointCloud::loadArepoSnapshot(const std::string snapname, const std::array<MyFloat,6> newsubbox)
{
  std::cout << "Loading " << snapname << "...\n";

  HighFive::File file(snapname, HighFive::File::ReadOnly);

  //Header
  HighFive::Group header = file.getGroup("Header");
  //Get boxsize
  MyFloat boxsize;
  header.getAttribute("BoxSize").read(boxsize);
  //Get current number of particles
  std::vector<size_t> npart_all_types;
  header.getAttribute("NumPart_Total").read(npart_all_types);
  size_t npart_temp = npart_all_types[0];

  //PartType0
  HighFive::Group part0 = file.getGroup("PartType0");
  //Get coordinates
  boost::multi_array<MyFloat, 2> pos_temp(boost::extents[npart_temp][3]);
  part0.getDataSet("Coordinates").read(pos_temp);
  //Get density
  std::vector<MyFloat> dens_temp;
  part0.getDataSet("Density").read(dens_temp);

  std::cout << "Applying bounding box...\n";

  subbox = newsubbox;
  //Find particles that are inside the (padded) frame
  MyFloat xmin = BOX_PAD_MIN * subbox[0];
  MyFloat xmax = BOX_PAD_MAX * subbox[1];
  MyFloat ymin = BOX_PAD_MIN * subbox[2];
  MyFloat ymax = BOX_PAD_MAX * subbox[3];
  MyFloat zmin = BOX_PAD_MIN * subbox[4];
  MyFloat zmax = BOX_PAD_MAX * subbox[5];

  std::vector<size_t> limit_idx;
  limit_idx.reserve(npart_temp);

  for(size_t i = 0; i < npart_temp; i++)
  {
    if((pos_temp[i][0] >= xmin) && (pos_temp[i][0] <= xmax) 
      && (pos_temp[i][1] >= ymin) && (pos_temp[i][1] <= ymax)
      && (pos_temp[i][2] >= zmin) && (pos_temp[i][2] <= zmax)) 
    {
      limit_idx.push_back(i);
    }
  }

  npart = limit_idx.size();

  //Load selected particles
  pts.resize(npart);
  dens.resize(npart);
  size_t idx;
  for(size_t i = 0; i < npart; i++) 
  { 
    idx = limit_idx[i];
    pts[i][0] = pos_temp[idx][0];
    pts[i][1] = pos_temp[idx][1];
    pts[i][2] = pos_temp[idx][2];
    dens[i] = dens_temp[idx]; 
  }

  std::cout << "npart: " << npart << "\n";

  std::cout << "Snapshot loaded." << std::endl;
  //Flag that tree is no longer up to date.
  tree_built = false;
}

void PointCloud::buildTree()
{
  if(npart <= 0)
  {
    std::cout << "There are no points in the cloud.\n";
    std::cout << "Aborting tree construction." << std::endl;
    return;
  }

  //Now build tree
  //reset here (vs make_unique) in case snap is reloaded
  std::cout << "Building tree...\n";
#ifdef TIMING_INFO
  auto start = std::chrono::high_resolution_clock::now();
#endif
  tree.reset(new my_kd_tree_t(3,*this,nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)));
  tree->buildIndex();
  tree_built = true;
#ifdef TIMING_INFO
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Tree build took " << duration.count() << " milliseconds." << std::endl;
#else
    std::cout << " Done." << std::endl;
#endif
}

size_t PointCloud::queryTree(const MyFloat query_pt[3]) const
{
  size_t result;
  MyFloat r2; //
  tree->knnSearch(&query_pt[0], 1, &result, &r2);
  return result;
}

size_t PointCloud::queryTree(const cartarr_t &query_pt) const
{
  //Need native array to pass to knnSearch
  MyFloat query_pt_native[3] = {query_pt[0], query_pt[1], query_pt[2]};
  size_t result;
  MyFloat r2; //
  tree->knnSearch(&query_pt_native[0], 1, &result, &r2);
  return result;
}
