#include <iostream>
using namespace std;
// virtual method implementation
struct VBase {
  void interface() {}
  virtual void implementation() { cout << "VBase" << endl; }
};
struct VDerived : public VBase {
  void implementation() override { cout << "VDerived" << endl; }
};

// crtp
template <typename Child> struct Base {
  void interface() { static_cast<Child *>(this)->implementation(); }
};

struct Derived : Base<Derived> {
  void implementation() { cerr << "Derived implementation\n"; }
};

template <typename ChildType> struct VectorBase {
  ChildType &underlying() { return static_cast<ChildType &>(*this); }
  inline ChildType &operator+=(const ChildType &rhs) {
    this->underlying() = this->underlying() + rhs;
    return this->underlying();
  }
};
struct Vec3f : public VectorBase<Vec3f> {
  float x{}, y{}, z{};
  Vec3f() = default;
  Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
};

inline Vec3f operator+(const Vec3f &lhs, const Vec3f &rhs) {
  Vec3f result;
  result.x = lhs.x + rhs.x;
  result.y = lhs.y + rhs.y;
  result.z = lhs.z + rhs.z;
  return result;
}
ostream &operator<<(ostream &os, const Vec3f &rhs) {
  return os << rhs.x << ", " << rhs.y << ", " << rhs.z << endl;
}

int main() {
  // runtime polymorphism
  VBase *v = new VDerived();
  v->implementation();

  // static polymorphism
  Derived derive;
  derive.implementation();

  Vec3f vec1(1, 2, 3), vec2(2, 3, 4);
  vec1 += vec2;
  cout << vec1 << vec2;
  return 0;
}