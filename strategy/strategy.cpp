/*================================================================================                               
*     Source: ../RCM/StrategyStudio/examples/strategies/SimpleMomentumStrategy/SimpleMomentumStrategy.cpp                                                        
*     Last Update: 2013/6/1 13:55:14                                                                            
*     Contents:                                     
*     Distribution:          
*                                                                                                                
*                                                                                                                
*     Copyright (c) RCM-X, 2011 - 2013.                                                  
*     All rights reserved.                                                                                       
*                                                                                                                
*     This software is part of Licensed material, which is the property of RCM-X ("Company"), 
*     and constitutes Confidential Information of the Company.                                                  
*     Unauthorized use, modification, duplication or distribution is strictly prohibited by Federal law.         
*     No title to or ownership of this software is hereby transferred.                                          
*                                                                                                                
*     The software is provided "as is", and in no event shall the Company or any of its affiliates or successors be liable for any 
*     damages, including any lost profits or other incidental or consequential damages relating to the use of this software.       
*     The Company makes no representations or warranties, express or implied, with regards to this software.                        
/*================================================================================*/   

#ifdef _WIN32
    #include "stdafx.h"
#endif

#include "strategy.h"

#include "FillInfo.h"
#include "AllEventMsg.h"
#include "ExecutionTypes.h"
#include <Utilities/Cast.h>
#include <Utilities/utils.h>

#include <math.h>
#include <iostream>
#include <cassert>

using namespace RCM::StrategyStudio;
using namespace RCM::StrategyStudio::MarketModels;
using namespace RCM::StrategyStudio::Utilities;

using namespace std;

Group7Strategy::Group7Strategy(StrategyID strategyID, const std::string& strategyName, const std::string& groupName):
    Strategy(strategyID, strategyName, groupName),
    m_momentum_map(),
	m_tradeprice_map(),
    m_instrument_order_id_map(),
    m_momentum(0),
    m_aggressiveness(0),
    m_position_size(100),
    m_debug_on(true),
    m_short_window_size(10),
    m_long_window_size(30),
	m_dia_last_trade_price(0),
	m_average_dia_ratio(0.0),
	m_num_dia_ratio_observations(0)
{
    //this->set_enabled_pre_open_data_flag(true);
    //this->set_enabled_pre_open_trade_flag(true);
    //this->set_enabled_post_close_data_flag(true);
    //this->set_enabled_post_close_trade_flag(true);
}

Group7Strategy::~Group7Strategy()
{
}

void Group7Strategy::OnResetStrategyState()
{
    m_momentum_map.clear();
    m_instrument_order_id_map.clear();
    m_momentum = 0;
}

void Group7Strategy::DefineStrategyParams()
{
    CreateStrategyParamArgs arg1("aggressiveness", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, m_aggressiveness);
    params().CreateParam(arg1);

    CreateStrategyParamArgs arg2("position_size", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_INT, m_position_size);
    params().CreateParam(arg2);

    CreateStrategyParamArgs arg3("short_window_size", STRATEGY_PARAM_TYPE_STARTUP, VALUE_TYPE_INT, m_short_window_size);
    params().CreateParam(arg3);
    
    CreateStrategyParamArgs arg4("long_window_size", STRATEGY_PARAM_TYPE_STARTUP, VALUE_TYPE_INT, m_long_window_size);
    params().CreateParam(arg4);
    
    CreateStrategyParamArgs arg5("debug", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_BOOL, m_debug_on);
    params().CreateParam(arg5);
}

void Group7Strategy::DefineStrategyCommands()
{
    StrategyCommand command1(1, "Reprice Existing Orders");
    commands().AddCommand(command1);

    StrategyCommand command2(2, "Cancel All Orders");
    commands().AddCommand(command2);
}

void Group7Strategy::RegisterForStrategyEvents(StrategyEventRegister* eventRegister, DateType currDate)
{    
    for (SymbolSetConstIter it = symbols_begin(); it != symbols_end(); ++it)
    {
        eventRegister->RegisterForBars(*it, BAR_TYPE_TIME, 10);
    }
}
void Group7Strategy::OnTrade(const TradeDataEventMsg& msg)
{
	std::cout << "OnTrade(): (" << msg.adapter_time() << "): " << msg.instrument().symbol() << ": " << msg.trade().size() << " @ $" << msg.trade().price(); // << std::endl;
	/*
	std::cout << "\t " <<
				msg.instrument().top_quote().bid_size() << " @ $"<< msg.instrument().top_quote().bid() <<
				msg.instrument().top_quote().ask_size() << " @ $"<< msg.instrument().top_quote().ask() <<
				std::endl;
	*/

	//add latest price to the map
	if ( msg.instrument().symbol().compare("DIA")!=0)
	{
		m_tradeprice_map[msg.instrument().symbol()] = msg.trade().price();
	}
	else
	{
		m_dia_last_trade_price = msg.trade().price();
	}
//	std::cout << "\t Dumping all symbols; has " << m_tradeprice_map.size() << "elements" << std::endl;


	double djia_index = 0;
	for (TradePriceMapIterator iter = m_tradeprice_map.begin(); iter!=m_tradeprice_map.end(); ++iter)
	{
//		std::cout << "\t\t" << (iter->first) << ": $" << iter->second << std::endl;
		djia_index += iter->second;
	}

	if (m_dia_last_trade_price!=0 and m_tradeprice_map.size()==31)
	{
		double ratio = djia_index / m_dia_last_trade_price;

		//update average ratio stats
		double new_average_ratio = ((m_average_dia_ratio*m_num_dia_ratio_observations) + ratio) / (m_num_dia_ratio_observations+1);


		std::cout << "; \t Dumping all symbols; " << // << m_tradeprice_map.size() << "elements; " <<
				"sum is " << djia_index <<
				" DIA last trade price is $" << m_dia_last_trade_price <<
				" ratio is " << ratio << //<< std::endl;
				" previous average ratio was " << m_average_dia_ratio <<
				" new average ratio is " << new_average_ratio;

		m_average_dia_ratio = new_average_ratio;
		m_num_dia_ratio_observations++;
	}
	std::cout << std::endl;

}

//TODO: need to fix the data source to get these three callbacks working

void Group7Strategy::OnTopQuote(const QuoteEventMsg& msg)
{
	std::cout << "OnTopQuote(): (" << msg.adapter_time() << "): " << msg.instrument().symbol() << ": " <<
		msg.instrument().top_quote().bid_size() << " @ $"<< msg.instrument().top_quote().bid() <<
		msg.instrument().top_quote().ask_size() << " @ $"<< msg.instrument().top_quote().ask() <<
		std::endl;

}

void Group7Strategy::OnQuote(const QuoteEventMsg& msg)
{
	std::cout << "OnQuote(): (" << msg.adapter_time() << "): " << msg.instrument().symbol() << ": " <<
			msg.instrument().top_quote().bid_size() << " @ $"<< msg.instrument().top_quote().bid() <<
			msg.instrument().top_quote().ask_size() << " @ $"<< msg.instrument().top_quote().ask() <<
			std::endl;
}

void Group7Strategy::OnDepth(const MarketDepthEventMsg& msg)
{
	std::cout << "OnDepth(): (" << msg.adapter_time() << "): " << msg.instrument().symbol() << ": " <<
				msg.instrument().top_quote().bid_size() << " @ $"<< msg.instrument().top_quote().bid() <<
				msg.instrument().top_quote().ask_size() << " @ $"<< msg.instrument().top_quote().ask() <<
				std::endl;
}


void Group7Strategy::OnOrderUpdate(const OrderUpdateEventMsg& msg)
{    
	std::cout << "OnOrderUpdate(): " << msg.update_time() << msg.name() << std::endl;
    if(msg.completes_order())
    {
		m_instrument_order_id_map[msg.order().instrument()] = 0;
		std::cout << "OnOrderUpdate(): order is complete; " << std::endl;
    }
}


void Group7Strategy::OnStrategyCommand(const StrategyCommandEventMsg& msg)
{
    switch (msg.command_id()) {
        case 1:
//            RepriceAll();
            break;
        case 2:
            trade_actions()->SendCancelAll();
            break;
        default:
            logger().LogToClient(LOGLEVEL_DEBUG, "Unknown strategy command received");
            break;
    }
}

