// util.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>

namespace py = pybind11;

/*** 辅助：把 obj.attr(name1) 或 obj.attr(name2) 取来的序列
 *    统一成 float64 的 contiguous NumPy array（若本来是 NumPy 且已连续，就零拷贝；否则一次性拷贝）。
 */
static py::array_t<double> get_array_attr_1d(const py::object& obj,
                                             const char* preferred,   // 比如 "s_np"/"a_np"
                                             const char* fallback)    // 比如 "s"/"a"
{
    auto np = py::module::import("numpy");
    py::object arr_obj;

    if (py::hasattr(obj, preferred)) {
        arr_obj = obj.attr(preferred);
    } else {
        arr_obj = obj.attr(fallback);
    }
    // ascontiguousarray 会在已是 C 连续且 dtype=float64 时零拷贝
    py::object f64 = np.attr("float64");
    py::object cca = np.attr("ascontiguousarray")(arr_obj, f64);
    return cca.cast<py::array_t<double>>();
}

/*** 取样条(一维)对象的 5 组系数数组视图：x/a/b/c/d
 *    约定：优先使用 *_np（零拷贝视图），否则回退到普通属性并一次性拷贝成 NumPy。
 */
struct SpBufs {
    py::array_t<double> x, a, b, c, d;   // 持有引用，保证内存有效
    const double* xp=nullptr; const double* ap=nullptr;
    const double* bp=nullptr; const double* cp=nullptr; const double* dp=nullptr;
    size_t n=0; // = x.size()
};

static SpBufs make_spbufs(const py::object& sp)
{
    SpBufs buf;
    buf.x = get_array_attr_1d(sp, "x_np", "x");
    buf.a = get_array_attr_1d(sp, "a_np", "a");
    buf.b = get_array_attr_1d(sp, "b_np", "b");
    buf.c = get_array_attr_1d(sp, "c_np", "c");
    buf.d = get_array_attr_1d(sp, "d_np", "d");

    buf.n  = buf.x.size();
    buf.xp = buf.x.data(); buf.ap = buf.a.data();
    buf.bp = buf.b.data(); buf.cp = buf.c.data(); buf.dp = buf.d.data();
    return buf;
}

/*** Python 里的 search_index(sp, x) —— 二分（与你给出的 Python 逻辑一致） */
static int search_index(const SpBufs& sp, double x)
{
    int min_i = 0;
    int max_i = static_cast<int>(sp.n) - 2;
    int index = -1;
    double count = 1.0;
    int mid_i = 0;

    if (max_i == -1) return 0;
    if (min_i == max_i) return 0;

    while (min_i <= max_i && count <= static_cast<double>(sp.n)) {
        count += 1.0;
        mid_i = (min_i + max_i) / 2;
        if (x < sp.xp[mid_i])       max_i = mid_i - 1;
        else if (x > sp.xp[mid_i])  min_i = mid_i + 1;
        else                        break;
    }
    index = mid_i;
    return index;
}

/*** Python calc(sp, t) —— 与你给出的 Python 版本逐句对齐 */
static inline double calc(const SpBufs& sp, double t)
{
    if (t < sp.xp[0])              return sp.ap[0];
    else if (t > sp.xp[sp.n-1])    return sp.ap[sp.n-1];
    else {
        int i  = search_index(sp, t);
        double dx = t - sp.xp[i];
        return sp.ap[i] + sp.bp[i]*dx + sp.cp[i]*dx*dx + sp.dp[i]*dx*dx*dx;
    }
}

/*** Python calcd(sp, t) —— 与你给出的 Python 版本逐句对齐 */
static inline double calcd(const SpBufs& sp, double t)
{
    if (t < sp.xp[0])              return sp.bp[0];
    else if (t > sp.xp[sp.n-1])    return sp.bp[sp.n-1];
    else {
        int i  = search_index(sp, t);
        double dx = t - sp.xp[i];
        return sp.bp[i] + 2.0*sp.cp[i]*dx + 3.0*sp.dp[i]*dx*dx;
    }
}

/*** Python calc_position(csp, s) —— 用 sx/sy 两条样条算 (x,y) */
static inline std::pair<double,double> calc_position(const SpBufs& sx, const SpBufs& sy, double s)
{
    return { calc(sx, s), calc(sy, s) };
}

/*** Python calc_yaw(csp, s) —— 与你给出的 Python 版本逐句对齐 */
static inline double calc_yaw(const SpBufs& sx, const SpBufs& sy, double s)
{
    double dx = calcd(sx, s);
    double dy = calcd(sy, s);
    double delta = 0.0;
    if (dx <= 0.0) {
        if (dy <= 0.0) { dx = -dx; dy = -dy; delta = -M_PI; }
        else           { delta =  M_PI; }
    }
    return std::atan(dy / dx) + delta;
}

/*** 公开给 Python 的入口：calc_cur_s(csp, ego_pose, index)
 *    注意：完全“对齐”你给的 Python 逻辑，包括那个 index 偏移的用法（不修复潜在 bug）。
 */
static py::tuple calc_cur_s_cpp(py::object csp, py::object ego_pose, int index)
{
    // 取 csp.s 数组；优先 s_np（零拷贝），否则把 s 转成 float64 C 连续数组
    py::array_t<double> s_arr = get_array_attr_1d(csp, "s_np", "s");
    const double* s_ptr = s_arr.data();
    size_t s_len = s_arr.size();

    // 取 csp.sx / csp.sy 并打包成只读视图
    py::object sx_obj = csp.attr("sx");
    py::object sy_obj = csp.attr("sy");
    SpBufs sx = make_spbufs(sx_obj);
    SpBufs sy = make_spbufs(sy_obj);

    // 取 ego_pose.x / y
    const double ego_x = py::float_(ego_pose.attr("x"));
    const double ego_y = py::float_(ego_pose.attr("y"));

    // 计算（释放 GIL，C++ 全速跑）
    double out_s = 0.01;
    int out_idx = 0;

    {
        py::gil_scoped_release release;

        // dist_mux
        std::vector<double> dist_mux;
        if (index < 0) index = 0;
        if (static_cast<size_t>(index) > s_len) index = static_cast<int>(s_len);
        dist_mux.reserve(s_len - index);

        for (size_t i = static_cast<size_t>(index); i < s_len; ++i) {
            double si = s_ptr[i];
            auto [gx, gy] = calc_position(sx, sy, si);
            double dx = ego_x - gx, dy = ego_y - gy;
            dist_mux.push_back(std::sqrt(dx*dx + dy*dy));
        }

        // 与 Python 对齐：min_index 是在 dist_mux 内的下标（**不加 index 偏移**）
        auto it = std::min_element(dist_mux.begin(), dist_mux.end());
        int min_index = static_cast<int>(std::distance(dist_mux.begin(), it));

        // 与 Python 对齐：直接用 csp.s[min_index]（注意，并没有 + index）
        double s_match = s_ptr[min_index];

        double yaw_match = calc_yaw(sx, sy, s_match);
        auto [x_match, y_match] = calc_position(sx, sy, s_match);

        double delta_x = ego_x - x_match;
        double delta_y = ego_y - y_match;
        double delta_s = delta_x * std::cos(yaw_match) + delta_y * std::sin(yaw_match);

        double s_val = s_match + delta_s;
        // round( ., 2 ) -> 四舍五入到 1/100
        s_val = std::round(s_val * 100.0) / 100.0;
        out_s = (s_val < 0.01) ? 0.01 : s_val;
        out_idx = min_index;
    }

    return py::make_tuple(out_s, out_idx);
}

// === C++ 版 calc_cur_d：与给定 Python 实现逐句对齐（不额外 round）===
static double calc_cur_d_cpp(py::object ego_pose, py::object csp, double cur_s)
{
    // 1) 取样条句柄，并做只读 NumPy 视图（零拷贝或一次性拷贝）
    py::object sx_obj = csp.attr("sx");
    py::object sy_obj = csp.attr("sy");
    SpBufs sx = make_spbufs(sx_obj);
    SpBufs sy = make_spbufs(sy_obj);

    // 2) 取 ego 坐标
    const double ex = py::float_(ego_pose.attr("x"));
    const double ey = py::float_(ego_pose.attr("y"));

    double cur_d = 0.0;

    {
        // 3) 计算（释放 GIL）
        py::gil_scoped_release release;

        // 参考点 & 航向
        auto [x_ref, y_ref] = calc_position(sx, sy, cur_s);
        double yaw_ref = calc_yaw(sx, sy, cur_s);

        // 偏移与符号
        double dx = ex - x_ref;
        double dy = ey - y_ref;
        double side = dy * std::cos(yaw_ref) - dx * std::sin(yaw_ref);
        double sgn  = (side > 0.0) ? 1.0 : ((side < 0.0) ? -1.0 : 0.0);

        cur_d = std::hypot(dx, dy) * sgn;
    }

    return cur_d;
}

// 1) 完全等价于：np.searchsorted(x, t, side='left') - 1
static inline int search_left_idx(const SpBufs& sp, double t)
{
    // xp 指向 float64 连续内存
    const double* begin = sp.xp;
    const double* end   = sp.xp + static_cast<ptrdiff_t>(sp.n);
    const double* it    = std::lower_bound(begin, end, t);
    // idx = it - begin - 1
    int idx = static_cast<int>(it - begin) - 1;
    // Python 里后续用的是 [i] 和 [i+1] 段，所以把 idx clamp 到 [0, n-2]
    if (idx < 0) idx = 0;
    int n2 = static_cast<int>(sp.n) - 2;
    if (idx > n2) idx = n2;
    return idx;
}

// 2) 对齐 Python Spline_.calc(t)
static inline double cs_calc(const SpBufs& sp, double t)
{
    if (t < sp.xp[0])            return sp.ap[0];
    if (t > sp.xp[sp.n - 1])     return sp.ap[sp.n - 1];

    int i = search_left_idx(sp, t);  // ← 等价于 np.searchsorted(...)-1
    double dx = t - sp.xp[i];
    return sp.ap[i] + sp.bp[i]*dx + sp.cp[i]*dx*dx + sp.dp[i]*dx*dx*dx;
}

// 3) 对齐 Python Spline_.calcd(t)
static inline double cs_calcd(const SpBufs& sp, double t)
{
    if (t < sp.xp[0])            return sp.bp[0];
    if (t > sp.xp[sp.n - 1])     return sp.bp[sp.n - 1];

    int i = search_left_idx(sp, t);
    double dx = t - sp.xp[i];
    return sp.bp[i] + 2.0*sp.cp[i]*dx + 3.0*sp.dp[i]*dx*dx;
}

// 计算 (x,y,z) —— 直接复用你已有的 calc()，确保与 Python Spline_.calc 一致
static inline std::tuple<double,double,double>
calc_position_xyz(const SpBufs& sx, const SpBufs& sy, const SpBufs& sz, double s)
{
    return { cs_calc(sx, s), cs_calc(sy, s), cs_calc(sz, s) };
}

// 5) 对齐 Python Spline3D.calc_yaw(s)
static inline double cs_calc_yaw(const SpBufs& sx, const SpBufs& sy, double s)
{
    double dx = cs_calcd(sx, s);
    double dy = cs_calcd(sy, s);
    double delta = 0.0;
    if (dx <= 0.0) {
        if (dy <= 0.0) { dx = -dx; dy = -dy; delta = -M_PI; }
        else           { delta =  M_PI; }
    }
    return std::atan(dy / dx) + delta;
}

// === C++ 版 calc_global_paths：与 Python 版本语义对齐（把路径从 Frenet 转到惯性系，并写回 fp.x/y/z/yaw）===
static py::list calc_global_paths_cpp(py::list fplist, py::object csp)
{
    // 取 csp.sx / sy / sz → 系数缓冲（零拷贝视图为主，fallback 一次性拷贝）
    py::object sx_obj = csp.attr("sx");
    py::object sy_obj = csp.attr("sy");
    py::object sz_obj = csp.attr("sz");
    SpBufs sx = make_spbufs(sx_obj);
    SpBufs sy = make_spbufs(sy_obj);
    SpBufs sz = make_spbufs(sz_obj);

    // 遍历每条 frenet path
    for (auto fp_obj : fplist) {
        py::object fp = py::reinterpret_borrow<py::object>(fp_obj);

        // 读 s / d（优先 *_np，退回到 list/ndarray 并拷到 float64 C 连续）
        py::array_t<double> s_arr = get_array_attr_1d(fp, "s_np", "s");
        py::array_t<double> d_arr = get_array_attr_1d(fp, "d_np", "d");
        const double* s_ptr = s_arr.data();
        const double* d_ptr = d_arr.data();
        size_t n = s_arr.size();
        if (d_arr.size() < n) {
            throw py::value_error("calc_global_paths: len(d) < len(s)");
        }
        if (n == 0) {
            continue;
        }

        // 结果先算进 C++ 向量，重型计算时释放 GIL
        std::vector<double> out_x(n), out_y(n), out_z(n), out_yaw(n);

        {
            py::gil_scoped_release release;

            for (size_t i = 0; i < n; ++i) {
                double s = s_ptr[i];

                // csp.calc_position(s) 等价：直接用样条系数在 C++ 里算
                auto [ix, iy, iz] = calc_position_xyz(sx, sy, sz, s);

                // csp.calc_yaw(s) 等价
                double iyaw = cs_calc_yaw(sx, sy, s);

                // 侧向偏移
                double di = d_ptr[i];

                // Frenet→惯性：以法向方向（yaw + pi/2）偏移 di
                out_x[i] = ix + di * std::cos(iyaw + M_PI / 2.0);
                out_y[i] = iy + di * std::sin(iyaw + M_PI / 2.0);
                out_z[i] = iz;
                out_yaw[i] = iyaw;
            }
        }

        // 把结果一次性写回 fp.x/y/z/yaw（用新 list 覆盖，等效于 Python 里先清空再 append）
        fp.attr("x")   = py::cast(out_x);
        fp.attr("y")   = py::cast(out_y);
        fp.attr("z")   = py::cast(out_z);
        fp.attr("yaw") = py::cast(out_yaw);
    }

    // 与 Python 版本一样返回 fplist（此时各 fp 已被就地改写）
    return fplist;
}

// ===== util.cpp 顶部（已有 include 之后）=====
#include <Eigen/Dense>

// 若 generate_single_frenet_path_cpp 在 calc_global_paths_cpp 之前使用：加前置声明
static py::list calc_global_paths_cpp(py::list fplist, py::object csp);

// ========= Frenet_path =========
struct Frenet_path {
    py::object id;  // None
    std::vector<double> t;
    std::vector<double> d, d_d, d_dd, d_ddd;
    std::vector<double> s, s_d, s_dd, s_ddd;
    double cd = 0.0, cv = 0.0, cf = 0.0;
    std::vector<double> x, y, z, yaw, ds, c;
    std::vector<double> v;
    Frenet_path() : id(py::none()) {}
};

// ========= Quintic / Quartic =========
struct QuinticPolynomial {
    double a0, a1, a2, a3, a4, a5;
    QuinticPolynomial(double xs, double vxs, double axs,
                      double xe, double vxe, double axe, double T)
    {
        a0 = xs; a1 = vxs; a2 = axs / 2.0;
        Eigen::Matrix3d A;
        A << std::pow(T,3), std::pow(T,4), std::pow(T,5),
             3*std::pow(T,2), 4*std::pow(T,3), 5*std::pow(T,4),
             6*T, 12*std::pow(T,2), 20*std::pow(T,3);
        Eigen::Vector3d b;
        b << (xe  - a0 - a1*T - a2*std::pow(T,2)),
             (vxe - a1 - 2*a2*T),
             (axe - 2*a2);
        Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
        a3 = x[0]; a4 = x[1]; a5 = x[2];
    }
    inline double calc_point(double t) const {
        return a0 + a1*t + a2*t*t + a3*std::pow(t,3) + a4*std::pow(t,4) + a5*std::pow(t,5);
    }
    inline double calc_first_derivative(double t) const {
        return a1 + 2*a2*t + 3*a3*std::pow(t,2) + 4*a4*std::pow(t,3) + 5*a5*std::pow(t,4);
    }
    inline double calc_second_derivative(double t) const {
        return 2*a2 + 6*a3*t + 12*a4*std::pow(t,2) + 20*a5*std::pow(t,3);
    }
    inline double calc_third_derivative(double t) const {
        return 6*a3 + 24*a4*t + 60*a5*std::pow(t,2);
    }
};

struct QuarticPolynomial {
    double a0, a1, a2, a3, a4;
    QuarticPolynomial(double xs, double vxs, double axs,
                      double vxe, double axe, double T)
    {
        a0 = xs; a1 = vxs; a2 = axs / 2.0;
        Eigen::Matrix2d A;
        A << 3*std::pow(T,2), 4*std::pow(T,3),
             6*T,            12*std::pow(T,2);
        Eigen::Vector2d b;
        b << (vxe - a1 - 2*a2*T),
             (axe - 2*a2);
        Eigen::Vector2d x = A.colPivHouseholderQr().solve(b);
        a3 = x[0]; a4 = x[1];
    }
    inline double calc_point(double t) const {
        return a0 + a1*t + a2*t*t + a3*std::pow(t,3) + a4*std::pow(t,4);
    }
    inline double calc_first_derivative(double t) const {
        return a1 + 2*a2*t + 3*a3*std::pow(t,2) + 4*a4*std::pow(t,3);
    }
    inline double calc_second_derivative(double t) const {
        return 2*a2 + 6*a3*t + 12*a4*std::pow(t,2);
    }
    inline double calc_third_derivative(double t) const {
        return 6*a3 + 24*a4*t;
    }
};

// ========= generate_single_frenet_path =========
static Frenet_path generate_single_frenet_path_cpp(py::tuple f_state,
                                                   double dt,
                                                   double df,
                                                   double Tf,
                                                   double Vf,
                                                   py::object csp)
{
    if (f_state.size() != 6)
        throw std::runtime_error("f_state must be (s, s_d, s_dd, d, d_d, d_dd)");

    const double s    = py::float_(f_state[0]);
    const double s_d  = py::float_(f_state[1]);
    const double s_dd = py::float_(f_state[2]);
    const double d    = py::float_(f_state[3]);
    const double d_d  = py::float_(f_state[4]);
    const double d_dd = py::float_(f_state[5]);

    if (dt <= 0.0 || Tf <= 0.0)
        throw std::runtime_error("dt and Tf must be positive");

    // t = np.arange(0.0, Tf, dt)  (不含 Tf)
    std::vector<double> t;
    t.reserve(static_cast<size_t>(Tf / dt) + 2);
    for (double tt = 0.0; tt < Tf; tt += dt) t.push_back(tt);
    const size_t n = t.size();
    if (n == 0) throw std::runtime_error("Empty time grid");

    std::vector<double> v_d(n), v_d_d(n), v_d_dd(n), v_d_ddd(n);
    std::vector<double> v_s(n), v_s_d(n), v_s_dd(n), v_s_ddd(n);

    QuinticPolynomial  lat_qp(d, d_d, d_dd, df, 0.0, 0.0, Tf);
    QuarticPolynomial  lon_qp(s, s_d, s_dd, Vf, 0.0, Tf);

    {
        py::gil_scoped_release release;
        for (size_t i = 0; i < n; ++i) {
            const double tt = t[i];
            v_d[i]     = lat_qp.calc_point(tt);
            v_d_d[i]   = lat_qp.calc_first_derivative(tt);
            v_d_dd[i]  = lat_qp.calc_second_derivative(tt);
            v_d_ddd[i] = lat_qp.calc_third_derivative(tt);

            v_s[i]     = lon_qp.calc_point(tt);
            v_s_d[i]   = lon_qp.calc_first_derivative(tt);
            v_s_dd[i]  = lon_qp.calc_second_derivative(tt);
            v_s_ddd[i] = lon_qp.calc_third_derivative(tt);
        }
    }

    Frenet_path fp;
    fp.t     = std::move(t);
    fp.d     = std::move(v_d);
    fp.d_d   = std::move(v_d_d);
    fp.d_dd  = std::move(v_d_dd);
    fp.d_ddd = std::move(v_d_ddd);
    fp.s     = std::move(v_s);
    fp.s_d   = std::move(v_s_d);
    fp.s_dd  = std::move(v_s_dd);
    fp.s_ddd = std::move(v_s_ddd);

    // 等价于 Python: fp = self.calc_global_paths([fp])[0]
    py::list lst;
    lst.append(py::cast(fp));
    py::list out = calc_global_paths_cpp(lst, csp);
    Frenet_path fp_done = out[0].cast<Frenet_path>();
    return fp_done;
}

// ===== util.cpp 顶部（已有 include 之后）=====
#include <numeric>
#include <limits>
#include <memory>

/* ---------- 1D Cubic Spline == Python Spline_ ---------- */
class Spline_ {
public:
    Spline_(const std::vector<double>& x, const std::vector<double>& y)
        : nx_(x.size()), x_(x), a_(y), b_(nx_-1), c_(nx_), d_(nx_-1) {
        computeCoefficients();
    }

    // 位置（与 Python 一致：区间外取端点值）
    double calc(double t) const {
        if (t < x_.front()) return a_.front();
        if (t > x_.back())  return a_.back();
        std::size_t i = segment(t);
        double dx = t - x_[i];
        // 连乘避免 pow
        return (((d_[i]*dx + c_[i])*dx + b_[i])*dx + a_[i]);
    }

    // 一阶导（区间外取端点 b）
    double calcd(double t) const {
        if (t < x_.front()) return b_.front();
        if (t > x_.back())  return b_.back();
        std::size_t i = segment(t);
        double dx = t - x_[i];
        return (3.0*d_[i]*dx + 2.0*c_[i])*dx + b_[i];
    }

    // 二阶导（区间外：返回 NaN，由 pybind 包装层转 None）
    double calcdd_raw(double t) const {
        if (t < x_.front() || t > x_.back()) return std::numeric_limits<double>::quiet_NaN();
        std::size_t i = segment(t);
        double dx = t - x_[i];
        return 6.0*d_[i]*dx + 2.0*c_[i];
    }

    // 三阶导（区间外：返回 NaN，由 pybind 包装层转 None）
    double calcddd_raw(double t) const {
        if (t < x_.front() || t > x_.back()) return std::numeric_limits<double>::quiet_NaN();
        return 6.0 * d_[segment(t)];
    }

    // 零拷贝视图所需
    const std::vector<double>& x() const { return x_; }
    const std::vector<double>& a() const { return a_; }
    const std::vector<double>& b() const { return b_; }
    const std::vector<double>& c() const { return c_; }
    const std::vector<double>& d() const { return d_; }

private:
    std::size_t nx_;
    std::vector<double> x_, a_, b_, c_, d_;

    // 等价 np.searchsorted(...,'left')-1 并 clamp 到 [0,nx_-2]
    std::size_t segment(double t) const {
        auto pos = std::lower_bound(x_.begin(), x_.end(), t) - x_.begin();
        if (pos == 0) return 0;
        std::size_t idx = pos - 1;
        return (idx > nx_-2) ? (nx_-2) : idx;
    }

    void computeCoefficients() {
        std::vector<double> h(nx_-1);
        for (std::size_t i=0;i<nx_-1;++i) h[i] = x_[i+1]-x_[i];

        // A c = B  （完全按 Python 版构造）
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nx_, nx_);
        A(0,0) = 1.0; A(nx_-1,nx_-1) = 1.0;
        for (std::size_t i=0; i<nx_-2; ++i) {
            A(i+1,i)   = h[i];
            A(i+1,i+1) = 2.0*(h[i]+h[i+1]);
            A(i,i+1)   = h[i];
        }
        Eigen::VectorXd B = Eigen::VectorXd::Zero(nx_);
        for (std::size_t i=1;i<nx_-1;++i) {
            B(i) = 3.0*(a_[i+1]-a_[i])  / h[i]
                 - 3.0*(a_[i]  -a_[i-1])/ h[i-1];
        }

        Eigen::VectorXd c_eig = A.ldlt().solve(B);
        for (std::size_t i=0;i<nx_;++i) c_[i] = c_eig(i);

        for (std::size_t i=0;i<nx_-1;++i) {
            d_[i] = (c_[i+1]-c_[i])/(3.0*h[i]);
            b_[i] = (a_[i+1]-a_[i])/h[i] - h[i]*(c_[i+1]+2.0*c_[i])/3.0;
        }
    }
};

/* ---------- 3D Cubic Spline == Python Spline3D ---------- */
class Spline3D {
public:
    Spline3D(const std::vector<double>& x,
             const std::vector<double>& y,
             const std::vector<double>& z)
    {
        const std::size_t n = x.size();
        s_.reserve(n);
        s_.push_back(0.0);
        for (std::size_t i=1;i<n;++i) {
            const double ds = std::sqrt((x[i]-x[i-1])*(x[i]-x[i-1])
                                       +(y[i]-y[i-1])*(y[i]-y[i-1])
                                       +(z[i]-z[i-1])*(z[i]-z[i-1]));
            s_.push_back(s_.back()+ds);
        }
        sx_ = std::make_unique<Spline_>(s_, x);
        sy_ = std::make_unique<Spline_>(s_, y);
        sz_ = std::make_unique<Spline_>(s_, z);
    }

    std::tuple<double,double,double> calc_position(double s) const {
        return { sx_->calc(s), sy_->calc(s), sz_->calc(s) };
    }

    // 与你的 Python（delta+atan）完全一致
    double calc_yaw(double s) const {
        double dx = sx_->calcd(s);
        double dy = sy_->calcd(s);
        double delta = 0.0;
        if (dx <= 0.0) {
            if (dy <= 0.0) { dx = -dx; dy = -dy; delta = -M_PI; }
            else           { delta =  M_PI; }
        }
        return std::atan(dy/dx) + delta;
    }

    // 按 Python 3D 版本（分母一阶）对齐
    double calc_curvature(double s) const {
        const double dx  = sx_->calcd(s);
        const double dy  = sy_->calcd(s);
        const double ddx = sx_->calcdd_raw(s);
        const double ddy = sy_->calcdd_raw(s);
        return (ddy*dx - ddx*dy) / (dx*dx + dy*dy);
    }

    double calc_pitch(double s) const {
        const double dx = sx_->calcd(s);
        const double dz = sz_->calcd(s);
        return std::atan2(dz, dx);
    }

    // 暴露零拷贝、嵌套求值
    const std::vector<double>& s() const { return s_; }
    Spline_& sx() const { return *sx_; }
    Spline_& sy() const { return *sy_; }
    Spline_& sz() const { return *sz_; }

private:
    std::vector<double> s_;
    std::unique_ptr<Spline_> sx_, sy_, sz_;
};

// ---- helper：把 std::vector<double> 暴露为零拷贝 np.ndarray，并绑定 base 以延长生命周期
auto vec_to_np = [](std::vector<double>& v, py::object base) {
    return py::array_t<double>(
        { static_cast<py::ssize_t>(v.size()) },       // shape
        { static_cast<py::ssize_t>(sizeof(double)) }, // stride
        v.data(),
        base
    );
};

PYBIND11_MODULE(local_utils_cpp, m) {
    m.doc() = "calc_cur_s / calc_cur_d (C++ core; consuming Python csp/ego via NumPy views)";

    m.def("calc_cur_s",
          &calc_cur_s_cpp,
          py::arg("csp"), py::arg("ego_pose"), py::arg("index"),
          "Compute (s, min_index) exactly matching the given Python logic.\n"
          "csp must have attributes: s or s_np, sx, sy; and sx/sy must have x/a/b/c/d (or *_np).");

    m.def("calc_cur_d",
          &calc_cur_d_cpp,
          py::arg("ego_pose"), py::arg("csp"), py::arg("cur_s"),
          "Compute signed lateral offset d at cur_s using the given Python logic.");

    m.def("calc_global_paths",
      &calc_global_paths_cpp,
      py::arg("fplist"), py::arg("csp"),
      "Transform a list of frenet paths into global (x,y,z,yaw) using csp.sx/sy/sz.\n"
      "This function overwrites fp.x/fp.y/fp.z/fp.yaw for each path and returns the same fplist.");

    // 导出 Frenet_path（字段与 Python 一一对应）
    py::class_<Frenet_path>(m, "Frenet_path")
        .def(py::init<>())
        .def_readwrite("id",   &Frenet_path::id)
        .def_readwrite("t",    &Frenet_path::t)
        .def_readwrite("d",    &Frenet_path::d)
        .def_readwrite("d_d",  &Frenet_path::d_d)
        .def_readwrite("d_dd", &Frenet_path::d_dd)
        .def_readwrite("d_ddd",&Frenet_path::d_ddd)
        .def_readwrite("s",    &Frenet_path::s)
        .def_readwrite("s_d",  &Frenet_path::s_d)
        .def_readwrite("s_dd", &Frenet_path::s_dd)
        .def_readwrite("s_ddd",&Frenet_path::s_ddd)
        .def_readwrite("cd",   &Frenet_path::cd)
        .def_readwrite("cv",   &Frenet_path::cv)
        .def_readwrite("cf",   &Frenet_path::cf)
        .def_readwrite("x",    &Frenet_path::x)
        .def_readwrite("y",    &Frenet_path::y)
        .def_readwrite("z",    &Frenet_path::z)
        .def_readwrite("yaw",  &Frenet_path::yaw)
        .def_readwrite("ds",   &Frenet_path::ds)
        .def_readwrite("c",    &Frenet_path::c)
        .def_readwrite("v",    &Frenet_path::v);

    // 导出生成函数：与 Python 版 generate_single_frenet_path 同参
    m.def("generate_single_frenet_path",
        &generate_single_frenet_path_cpp,
        py::arg("f_state"),
        py::arg("dt"),
        py::arg("df") = 0.0,
        py::arg("Tf") = 4.0,
        py::arg("Vf") = 30.0 / 3.6,
        py::arg("csp"),
        "Generate one Frenet_path with (s,s_d,s_dd,d,d_d,d_dd), time step dt, "
        "target df/Tf/Vf, and map to global using csp (Spline3D-like: sx/sy/sz).");

    // ---------- Spline_ ----------
    py::class_<Spline_>(m, "Spline_")
    .def(py::init<const std::vector<double>&, const std::vector<double>&>())
    .def("calc",  &Spline_::calc)
    .def("calcd", &Spline_::calcd)
    // 越界行为：返回 None（内部 NaN → None）
    .def("calcdd", [](Spline_& s, double t)->py::object {
        double v = s.calcdd_raw(t);
        if (std::isnan(v)) return py::none();
        return py::float_(v);
    })
    .def("calcddd", [](Spline_& s, double t)->py::object {
        double v = s.calcddd_raw(t);
        if (std::isnan(v)) return py::none();
        return py::float_(v);
    })
    // 零拷贝属性（与 SpBufs 路径一致）
    .def_property_readonly("x_np", [](Spline_& s){ return vec_to_np(const_cast<std::vector<double>&>(s.x()), py::cast(&s)); })
    .def_property_readonly("a_np", [](Spline_& s){ return vec_to_np(const_cast<std::vector<double>&>(s.a()), py::cast(&s)); })
    .def_property_readonly("b_np", [](Spline_& s){ return vec_to_np(const_cast<std::vector<double>&>(s.b()), py::cast(&s)); })
    .def_property_readonly("c_np", [](Spline_& s){ return vec_to_np(const_cast<std::vector<double>&>(s.c()), py::cast(&s)); })
    .def_property_readonly("d_np", [](Spline_& s){ return vec_to_np(const_cast<std::vector<double>&>(s.d()), py::cast(&s)); })
    // 兼容回退：也提供 Python list 版
    .def_property_readonly("x", [](Spline_& s){ return py::cast(s.x()); })
    .def_property_readonly("a", [](Spline_& s){ return py::cast(s.a()); })
    .def_property_readonly("b", [](Spline_& s){ return py::cast(s.b()); })
    .def_property_readonly("c", [](Spline_& s){ return py::cast(s.c()); })
    .def_property_readonly("d", [](Spline_& s){ return py::cast(s.d()); });

    // ---------- Spline3D ----------
    py::class_<Spline3D>(m, "Spline3D")
    .def(py::init<const std::vector<double>&, const std::vector<double>&, const std::vector<double>&>())
    .def("calc_position", &Spline3D::calc_position)
    .def("calc_yaw",      &Spline3D::calc_yaw)
    .def("calc_curvature",&Spline3D::calc_curvature)
    .def("calc_pitch",    &Spline3D::calc_pitch)
    .def_property_readonly("s_np", [](Spline3D& sp){
        auto &s = const_cast<std::vector<double>&>(sp.s());
        return vec_to_np(s, py::cast(&sp));
    })
    .def_property_readonly("s", [](Spline3D& sp){ return py::cast(sp.s()); })
    // 让 sx/sy/sz 在 Python 侧继续当对象用；生命周期绑定到 Spline3D
    .def_property_readonly("sx", [](Spline3D& sp)->Spline_& { return sp.sx(); },
                           py::return_value_policy::reference_internal)
    .def_property_readonly("sy", [](Spline3D& sp)->Spline_& { return sp.sy(); },
                           py::return_value_policy::reference_internal)
    .def_property_readonly("sz", [](Spline3D& sp)->Spline_& { return sp.sz(); },
                           py::return_value_policy::reference_internal);
}

