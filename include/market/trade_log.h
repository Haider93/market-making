#ifndef MARKET_TRADE_LOG_H
#define MARKET_TRADE_LOG_H

#include <list>
#include <memory>
#include "../market/order.h"

namespace market {

class TradeLog
{
    private:
		std::list<unique_ptr<Order>>;

};

}

#endif
