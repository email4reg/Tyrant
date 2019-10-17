#include<stdio.h>

#define NL '\n'
#define PI 3.1415926
#define S(r) PI * r * r
#define C(r) 2 * PI * r
#define FANQIE 3.7
#define JIELAN 7
#define XIQIN 1.3
#define KONGXINCAI 8
#define YANGCONG 2.4
#define YOUCAI 9
#define HUANGGUA 6.3
#define BAILUOBO 0.5

int func1()
{
    printf("Line1%c",NL);
    printf("Line2%c",NL);

    return 0;
}

int func2()
{
    float r;
    printf("请输入半径大小:");
    scanf("%f",&r);
    printf("半径为%.2f,面积为%.2f,周长为%.2f\n",r,S(r),C(r));

    return 0;
}

int func3()
{
    float price;
    price = 0.5 * (2 * FANQIE + KONGXINCAI + YOUCAI);
    printf("小明需要支付%.2f\n", price);

    price = 0.5 * (3 * XIQIN + 0.5 * YANGCONG + 5 * HUANGGUA);
    printf("小红需要支付%.2f\n", price);

    price = 0.5 * (10 * HUANGGUA + 20 * BAILUOBO);
    printf("小甲鱼需要支付%.2f\n", price);

    return 0;
}


int main()
{
    // func1();
    // func2();
    func3();

    return 0;
}

