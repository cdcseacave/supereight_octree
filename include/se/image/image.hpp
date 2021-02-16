#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <cassert>
#include <Eigen/StdVector>

namespace se {

  template <typename T>
    class Image {

      public:
        Image() {
        }
        Image(const unsigned w, const unsigned h) {
          resize(w, h);
        }
        Image(const unsigned w, const unsigned h, const T& val) {
          resize(w, h, val);
        }

        void resize(const unsigned w, const unsigned h) {
          width_ = w; height_ = h;
          assert(width_ > 0 && height_ > 0);
          data_.resize(width_ * height_);
        }
        void resize(const unsigned w, const unsigned h, const T& val) {
          width_ = w; height_ = h;
          assert(width_ > 0 && height_ > 0);
          data_.resize(width_ * height_, val);
        }

        T&       operator[](std::size_t idx)       { return data_[idx]; }
        const T& operator[](std::size_t idx) const { return data_[idx]; }

        T&       operator()(const int x, const int y)       { return data_[x + y*width_]; }
        const T& operator()(const int x, const int y) const { return data_[x + y*width_]; }

        bool        empty()  const { return data_.empty(); };
        std::size_t size()   const { return width_ * height_; };
        int         width () const { return width_;  };
        int         height() const { return height_; };

        T* data()             { return data_.data(); }
        const T* data() const { return data_.data(); }

      private:
        int width_, height_;
        std::vector<T, Eigen::aligned_allocator<T> > data_;
    };

} // end namespace se
#endif
