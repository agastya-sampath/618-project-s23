// Reference : Assignment 4 code for Vec2 struct
typedef struct rgb
{
    float r;
    float g;
    float b;
    rgb(float vr = 0.0f, float vg = 0.0f, float vb = 0.0f) : r(vr), g(vg), b(vb) {}
    
    static float dot(rgb v0, rgb v1) { return v0.r * v1.r + v0.g * v1.g + v0.b * v1.b; }
    float &operator[](int i) { return ((float *)this)[i]; }

    rgb operator*(float s) const { return rgb(*this) *= s; }
    rgb operator*(rgb vin) const { return rgb(*this) *= vin; }
    rgb &operator*=(float s)
    {
        r *= s;
        g *= s;
        b *= s;
        return *this;
    }
    rgb &operator*=(rgb vin)
    {
        r *= vin.r;
        g *= vin.g;
        b *= vin.b;
        return *this;
    }

    rgb operator+(rgb vin) const { return rgb(*this) += vin; }
    rgb operator-(rgb vin) const { return rgb(*this) -= vin; }
    rgb operator-() const { return rgb(-r, -g, -b); }
    rgb &operator+=(rgb vin)
    {
        r += vin.r;
        g += vin.g;
        b += vin.b;
        return *this;
    }
    rgb &operator-=(rgb vin)
    {
        r -= vin.r;
        g -= vin.g;
        b -= vin.b;
        return *this;
    }
    
    rgb &operator=(float v)
    {
        r = g = b = v;
        return *this;
    }
    
} rgb_t;

bool compareR(const rgb_t &a, const rgb_t &b) {
    return a.r < b.r;
}

bool compareG(const rgb_t &a, const rgb_t &b) {
    return a.g < b.g;
}

bool compareB(const rgb_t &a, const rgb_t &b) {
    return a.b < b.b;
}