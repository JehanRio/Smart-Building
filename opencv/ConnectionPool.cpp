#include "ConnectionPool.h"



ConnectionPool* ConnectionPool::getConnectPool()
{
	//static ConnectionPool pool; // 静态局部对象：第二次调用时已经存在，得到的是同一块内存地址
	return (new ConnectionPool("root", "ljh307999", "test", "127.0.0.1", 3306));
}

ConnectionPool::ConnectionPool(string user, string passwd, string dbName, string ip, unsigned short port = 3306)
{
	m_user = user;
	m_passwd=passwd;
	m_dbName = dbName;
	m_ip = ip;
	m_port = port;

	for (int i = 0; i < m_minSize; i++)
	{
		addConnection();    // 添加连接
	}
	thread producer(&ConnectionPool::produceConnection, this);   // 生产 std::thread 调用类的成员函数需要传递类的一个对象指针作为参数
	thread recycler(&ConnectionPool::recycleConnection, this);   // 销毁
	producer.detach();
	recycler.detach();  // 子线程分离
}

shared_ptr<MysqlConn> ConnectionPool::getConnection()
{
	unique_lock<mutex> locker(m_mutexQ);
	while (m_connectionQ.empty())
	{
		if (cv_status::timeout == m_cond.wait_for(locker, chrono::milliseconds(m_timeout)));  // wait一段时间 ms
		{
			// 双重检查，说明任务队列依旧为空
			if (m_connectionQ.empty())  // 感觉不如直接用wait
			{
				//return nullptr;
				continue;
			}
		}
	}
	shared_ptr<MysqlConn> connptr(m_connectionQ.front(), [this](MysqlConn* conn)    // 第二个参数：指定删除器 （重定义）
		{
			lock_guard<mutex> locker(m_mutexQ); // 当locker被析构时，先进行自动解锁，再析构
			//m_mutexQ.lock();
			conn->refreshAliveTime();
			m_connectionQ.push(conn);   // 回收,共享资源需要加锁
			//m_mutexQ.unlock();
		});
	m_connectionQ.pop();
	m_cond.notify_all();    // 唤醒生产者,对消费者没有影响（上面代码重复执行一次循环）
	return connptr;
}

ConnectionPool::~ConnectionPool()
{
	while (!m_connectionQ.empty())
	{
		MysqlConn* conn = m_connectionQ.front();
		m_connectionQ.pop();
		delete conn;
	}
}

void ConnectionPool::produceConnection()
{
	while (true)
	{
		unique_lock<mutex> locker(m_mutexQ);    // 包装了一个互斥锁对象，由locker对象自动管理开锁解锁
		while (m_connectionQ.size() >= m_minSize)   // 加while循环而不是if
		{
			m_cond.wait(locker);
		}
		addConnection();
		m_cond.notify_all();    // 唤醒消费者,如果唤醒了其他生产者，则生产者会阻塞在第二轮while循环
	}
}

void ConnectionPool::recycleConnection()
{
	while (true)
	{
		this_thread::sleep_for(chrono::seconds(1)); // 休眠1s
		lock_guard<mutex> locker(m_mutexQ);
		while (m_connectionQ.size() > m_minSize)    // 只有大于最小数才需要销毁
		{
			MysqlConn* conn = m_connectionQ.front();    // 取出队头，因为队头时间最长
			if (conn->getAliveTime() >= m_maxIdleTime)
			{
				m_connectionQ.pop();
				delete conn;
			}
			else
			{
				break;  // 什么也不做
			}
		}
	}
}

void ConnectionPool::addConnection()
{
	MysqlConn* conn = new MysqlConn;
	if (!conn->connect(m_user, m_passwd, m_dbName, m_ip, m_port))
	{
		cout << "连接失败！" << endl;
		exit(0);
	}
	conn->refreshAliveTime();   // 刷新空闲时间
	m_connectionQ.push(conn);
}
