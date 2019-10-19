#include <stdio.h>
#include <math.h>


int func1()
{
    float price,area,yir,mir; //单价、面积、年利率、月利率
    float interest,loan; // 利息、贷款总额
    float ave_repay,down_payment; // 月均还款、首期付款
    float total_price,total_repay; // 房款总额、还款总额
    int ratio,time; // 按揭成数、按揭年数

    printf("请输入单价(元/平方):");
    scanf("%f",&price);
    printf("请输入面积:");
    scanf("%f",&area);
    printf("请输入按揭成数:");
    scanf("%d",&ratio);
    printf("请输入按揭年数:");
    scanf("%d",&time);
    printf("请输入当前基准年利率:");
    scanf("%f",&yir);

    printf("========报告结果========\n");
    mir = yir / 100 / 12;
    time = time * 12;
    total_price = price * area; // 房款总额
    loan = total_price  * ratio / 10; // 贷款总额
    ave_repay = loan * mir * pow((1 + mir), time) / (pow((1 + mir), time) - 1); // 月均还款
    interest = time * ave_repay - loan; // 支付利息
    total_repay = interest + loan;
    down_payment = total_price * (1 - (float)ratio / 10); // 首期付款

    printf("房款总额:%.2f元\n",total_price);
    printf("首期付款:%.2f元\n",down_payment);
    printf("贷款总额:%.2f元\n",loan);
    printf("还款总额:%.2f元\n",total_repay);
    printf("支付利息:%.2f元\n",interest);
    printf("月均还款:%.2f元\n",ave_repay);

    return 0;
}

int func2()
{
    float p;

    p = 10000 * (1 + 2.75 / 100 * 5);
    printf("一次性定期存5年的本息和为:%.2f元\n",p);

    p = 10000 * (1 + 2.75 / 100 * 3) * (1 + 2.25 / 100 * 2);
    printf("先存3年定期，到期后本息再存2年定期:%.2f元\n", p);

    p = 10000 * pow(1 + 1.75 / 100,5);
    printf("存一年定期，到期后本息再存一年定期，连续存5次:%.2f元\n", p);

    return 0;
}

int main()
{
    // func1();
    func2();

    return 0;
}