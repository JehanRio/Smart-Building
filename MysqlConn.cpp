#include "MysqlConn.h"

MysqlConn::MysqlConn()
{
	m_conn = mysql_init(nullptr);
	mysql_set_character_set(m_conn, "utf8");  // 设置编码
}

MysqlConn::~MysqlConn()
{
	if (m_conn != nullptr)
	{
		mysql_close(m_conn);
	}
	freeResult();
}

bool MysqlConn::connect(string user, string passwd, string dbName, string ip, unsigned short port)
{
	MYSQL* ptr = mysql_real_connect(m_conn, ip.c_str(), user.c_str(), passwd.c_str(), dbName.c_str(), port, nullptr, 0);	// 转换成char*类型
	return ptr != nullptr;	// 如果为空，则为false
}

bool MysqlConn::update(string sql)
{
	if (mysql_query(m_conn, sql.c_str()))	// 返回0则执行成功
	{
		return false;
	}
	return true;
}

bool MysqlConn::query(string sql)
{
	freeResult();
	if (mysql_query(m_conn, sql.c_str()))
	{
		return false; 
	}
	// 使用mysql_query来发送SQL语句，然后使用mysql_store_result或mysql_use_result来提取数据
	m_result = mysql_store_result(m_conn);	// 在mysql客户端得到结果集
	return true;
}

bool MysqlConn::next()
{
	if (m_result != nullptr)
	{
		m_row = mysql_fetch_row(m_result);
		if (m_row != nullptr)
			return true;
	}
	return false;
}

string MysqlConn::Value(int index)
{
	int columnCount = mysql_num_fields(m_result);
	if (index >= columnCount || index < 0)
	{
		return string();
	}
	char* val = m_row[index];
	unsigned long length = mysql_fetch_lengths(m_result)[index];
	return string(val,length);
}

bool MysqlConn::transaction()
{
	return mysql_autocommit(m_conn,false);	// 手动提交
}

bool MysqlConn::commit()
{
	return mysql_commit(m_conn);
}

bool MysqlConn::rollback()
{
	return mysql_rollback(m_conn);
}

void MysqlConn::refreshAliveTime()
{
	m_alivetime = steady_clock::now();
}

long long MysqlConn::getAliveTime()
{
	nanoseconds res = steady_clock::now() - m_alivetime;	// 高精度:纳秒
	milliseconds millsec = duration_cast<milliseconds>(res);	// 高精度->低精度：毫秒
	return millsec.count();	// 得到的时间间隔里面有多少个毫秒
}

void MysqlConn::freeResult()
{
	if (m_result)
	{
		mysql_free_result(m_result);
		m_result = nullptr;
	}
}
