#ifndef MARKET_TARGET_PRICE_H
#define MARKET_TARGET_PRICE_H

#include "../market/book.h"
#include "../utilities/accumulators.h"


namespace market {
namespace tp {

class TargetPrice
{
    protected:
        double val_;

    public:
        TargetPrice();
        virtual ~TargetPrice() = default;

        double get();

        virtual bool ready();
        virtual void update(market::AskBook<>& ab, market::BidBook<>& bb);
        virtual void clear();
		virtual void set(double tp);//set val_
		virtual double avg();
};

class MidPrice: public TargetPrice
{
    private:
        RollingMean<double> mp_;

    public:
        MidPrice(int lookback);

        bool ready();
        void update(market::AskBook<>& ab, market::BidBook<>& bb);
        void clear();
		double top();
		double avg();
		void set(double tp);
};

class MicroPrice: public TargetPrice
{
    private:
        RollingMean<double> mp_;

    public:
        MicroPrice(int lookback);

        bool ready();
        void update(market::AskBook<>& ab, market::BidBook<>& bb);
        void clear();
		void set(double tp);
		//double average();
};

class VWAP: public TargetPrice
{
    private:
        Accumulator<double> numerator_,
                            denominator_;

    public:
        VWAP(int lookback);

        bool ready();
        void update(market::AskBook<>& ab, market::BidBook<>& bb);
        void clear();
		void set(double tp);
		//double average();
};

}
}

#endif
