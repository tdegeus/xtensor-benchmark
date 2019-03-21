#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>

xt::xarray<int> S2(
  const std::vector<size_t>& roi,
  const xt::xarray<int>& f,
  const xt::xarray<int>& g,
  const xt::xarray<int>& fmask,
  const xt::xarray<int>& gmask)
{
  std::vector<size_t> mid = roi;

  for (auto& i: mid)
    i = (i-1)/2;

  xt::xarray<int> data = xt::zeros<int>(roi);

  for (size_t h = mid[0]; h < f.shape(0)-mid[0]; ++h) {
    for (size_t i = mid[1]; i < f.shape(1)-mid[1]; ++i) {
      for (size_t j = mid[2]; j < f.shape(2)-mid[2]; ++j) {

        if (fmask(h,i,j))
          continue;

        auto gi = xt::view(g,
          xt::range(h-mid[0], h+mid[0]+1),
          xt::range(i-mid[1], i+mid[1]+1),
          xt::range(j-mid[2], j+mid[2]+1));

        auto gmii = int(1) - xt::view(gmask,
          xt::range(h-mid[0], h+mid[0]+1),
          xt::range(i-mid[1], i+mid[1]+1),
          xt::range(j-mid[2], j+mid[2]+1));

        if (f(h,i,j) != 0)
          data += f(h,i,j) * gi * gmii;
      }
    }
  }

  return data;
}

xt::xarray<int> C2(
  const std::vector<size_t>& roi,
  const xt::xarray<int>& f,
  const xt::xarray<int>& g,
  const xt::xarray<int>& fmask,
  const xt::xarray<int>& gmask)
{
  std::vector<size_t> mid = roi;

  for (auto& i: mid)
    i = (i-1)/2;

  xt::xarray<int> data = xt::zeros<int>(roi);

  for (size_t h = mid[0]; h < f.shape(0)-mid[0]; ++h) {
    for (size_t i = mid[1]; i < f.shape(1)-mid[1]; ++i) {
      for (size_t j = mid[2]; j < f.shape(2)-mid[2]; ++j) {

        if (fmask(h,i,j))
          continue;

        auto gi = xt::view(g,
          xt::range(h-mid[0], h+mid[0]+1),
          xt::range(i-mid[1], i+mid[1]+1),
          xt::range(j-mid[2], j+mid[2]+1));

        auto gmii = int(1) - xt::view(gmask,
          xt::range(h-mid[0], h+mid[0]+1),
          xt::range(i-mid[1], i+mid[1]+1),
          xt::range(j-mid[2], j+mid[2]+1));

        if (f(h,i,j) != 0)
          data += xt::where(xt::equal(f(h,i,j), gi), gmii, int(0));
      }
    }
  }

  return data;
}

int main()
{
  xt::xarray<double> r = 2. * xt::random::randn<double>({1,501,501});
  xt::xarray<int> I = xt::where(xt::greater(r, 1.), int(1), int(0));
  xt::xarray<int> mask = xt::zeros<int>(I.shape());

  xt::xarray<int> data_S2 = S2({1,11,11}, I, I, mask, mask);
  xt::xarray<int> data_C2 = C2({1,11,11}, I, I, mask, mask);

  if (!(xt::all(xt::equal(data_S2, data_S2))))
    throw std::runtime_error("Result should be the same for binary input");

  return 0;
}
