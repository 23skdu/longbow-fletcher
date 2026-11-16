package client

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

// TLSConfig holds TLS/mTLS settings for Fletcher clients.
type TLSConfig struct {
	CertFile           string
	KeyFile            string
	CAFile             string
	ServerName         string
	InsecureSkipVerify bool
}

// ClientConfig configures a Fletcher Arrow Flight client.
type ClientConfig struct {
	TLS    *TLSConfig
	APIKey string
	Tenant string
}

type apiKeyCredentials struct {
	key    string
	tenant string
}

func (a apiKeyCredentials) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	md := map[string]string{}
	if a.key != "" {
		md["authorization"] = "Bearer " + a.key
	}
	if a.tenant != "" {
		md["x-tenant"] = a.tenant
	}
	return md, nil
}

func (a apiKeyCredentials) RequireTransportSecurity() bool { return false }

func buildDialOptions(cfg ClientConfig) ([]grpc.DialOption, error) {
	var opts []grpc.DialOption

	if cfg.TLS != nil {
		tlsCfg := &tls.Config{
			ServerName:         cfg.TLS.ServerName,
			InsecureSkipVerify: cfg.TLS.InsecureSkipVerify,
		}
		if cfg.TLS.CAFile != "" {
			caPEM, err := os.ReadFile(cfg.TLS.CAFile)
			if err != nil {
				return nil, fmt.Errorf("read CA file: %w", err)
			}
			pool := x509.NewCertPool()
			if !pool.AppendCertsFromPEM(caPEM) {
				return nil, fmt.Errorf("failed to parse CA certificate")
			}
			tlsCfg.RootCAs = pool
		}
		if cfg.TLS.CertFile != "" && cfg.TLS.KeyFile != "" {
			cert, err := tls.LoadX509KeyPair(cfg.TLS.CertFile, cfg.TLS.KeyFile)
			if err != nil {
				return nil, fmt.Errorf("load client key pair: %w", err)
			}
			tlsCfg.Certificates = []tls.Certificate{cert}
		}
		opts = append(opts, grpc.WithTransportCredentials(credentials.NewTLS(tlsCfg)))
	} else {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	if cfg.APIKey != "" || cfg.Tenant != "" {
		opts = append(opts, grpc.WithPerRPCCredentials(apiKeyCredentials{key: cfg.APIKey, tenant: cfg.Tenant}))
	}
	return opts, nil
}
