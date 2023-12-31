CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230207000000_e20230207235959_p20230208021505_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-08T02:15:05.398Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-07T00:00:00.000Z   time_coverage_end         2023-02-07T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxb��@  
�          A4��@����Fff@��HB*z�C��R@����Ǯ@�=qA�RC�ff                                    Bxb���  �          A5�@��R��@�\)B-p�C�9�@��R��  @��B �C�J=                                    Bxb�ތ            A3
=@ٙ��Dz�AQ�B?�
C��
@ٙ��У�@�{B ��C�
                                    Bxb��2  �          A0Q�@���C33@�33B6z�C�P�@�����@��\A�(�C�R                                    Bxb���  
�          A-�@޸R��p�@�
=A��C��
@޸R�33?L��@���C�+�                                    Bxb�
~  
�          A.�\@Ϯ��z�@�\)B{C���@Ϯ��R@Q�AJ{C��                                    Bxb�$  	�          A/
=@��
���\@\B��C��{@��
���@5Ar{C�K�                                    Bxb�'�  	s          A.�H@��R��=q@ƸRB{C�u�@��R�θR@i��A��HC�w
                                    Bxb�6p  
�          A+33@mp��33@�\)A��\C��=@mp���R�k���G�C�}q                                    Bxb�E  
�          A.ff@����\)@��RA���C��)@���33?�z�AG�C��=                                    Bxb�S�  
(          A.ff@�Q���@�{A�C��q@�Q��
=q?��@:=qC��H                                    Bxb�bb  �          A-��@�������@�(�A��C�'�@����	��?��@�(�C�U�                                    Bxb�q  
�          A,��@����\)@�33B=qC�5�@����=q@3�
As
=C�w
                                    Bxb��  �          A-p�A�
�\(�@�33B	Q�C��=A�
��33@vffA��HC�P�                                    Bxb��T  �          A-G�@�z���@�  A���C�H@�z�����@(��Ab�\C�j=                                    Bxb���  
�          A1Ap�����@�
=A�C�o\Ap���=q?�z�A!G�C�                                    Bxb���  "          A1A  ��Q�@��A���C��A  �θR@,��Ac\)C�aH                                    Bxb��F  
�          A/�@������\@��B	
=C��)@�����@Q�A�\)C��=                                    Bxb���  
�          A0(�A���ff@u�A�G�C��)A���33?k�@�G�C�p�                                    Bxb�ג  T          A/�
A\)��
=@\��A��\C�]qA\)���>�@=qC�L�                                    Bxb��8  
�          A-p�@�=q��G�@(�A<��C�@�=q��\)��\)��\C���                                    Bxb���  �          A*�R@����R@fffA�Q�C��@��� �ý�Q��C�P�                                    Bxb��  "          A*ff@�(�����@أ�B(�C�|)@�(����@�=qA�(�C���                                    Bxb�*  "          A,Q�@�����(�@�G�B;�C�xR@�����ff@�{A��HC��{                                    Bxb� �  
Z          A-�@��
��@��B?��C�C�@��
��@��RA癚C�G�                                    Bxb�/v  "          A,��@��
��
=@���B��C�n@��
�=q@�A333C�#�                                    Bxb�>  T          A,��@��\���@�33B�\C���@��\��
@E�A���C�=q                                    Bxb�L�  
�          A,Q�@�33�Ϯ@�ffB G�C��@�33�
=@O\)A��C���                                    Bxb�[h  
�          A*�H@��\���@�A܏\C�c�@��\�\)?333@q�C��                                    Bxb�j  A          A*�\@����@�p�AхC���@��p�>��@$z�C��                                    Bxb�x�  +          A*ff@�Q����@��A�\)C��q@�Q���?���@�  C�E                                    Bxb��Z  
(          A4��A)��>��@j�HA�z�@{A)����ff@\��A�\)C��H                                    Bxb��   
�          A4(�A&�H���@���A�\)C�nA&�H�?\)@?\)AvffC�H                                    Bxb���  �          A1�A���@��@�G�Ȁ\C�B�A����33@9��Ar�RC���                                    Bxb��L  
�          A0Q�A�\�mp�@��A���C�@ A�\��Q�@
=qA5��C�ff                                    Bxb���  �          A1��A�\����@]p�A��C�aHA�\����?��H@ȣ�C���                                    Bxb�И  "          A0  A�����?s33@��RC��\A���{��=q��C�'�                                    Bxb��>  T          A.�\A�����?���@���C�}qA���녿����C�w
                                    Bxb���  
�          A.�\Az����?\(�@��\C�}qAz����R������
C��q                                    Bxb���  
�          A,z�A����=q@W�A�
=C�}qA���ָR>�p�?��RC��                                    Bxb�0  6          A-A(���ff@<(�Ay�C�xRA(���G�?(�@L��C�s3                                    Bxb��  �          A.�RA{���?�z�Az�C��)A{���H�5�n�RC�3                                    Bxb�(|  "          A-�Az���  ?���@�=qC�>�Az����ÿ�Q��˅C�,�                                    Bxb�7"  �          A-��A\)���H@,(�Ae�C���A\)��  =���?�C�                                      Bxb�E�  T          A,��Ap����?k�@��C�(�Ap���p����H�
=C��R                                    Bxb�Tn  "          A*�HA����ff�G���
=C�
=A�����_\)���
C�ff                                    Bxb�c  
�          A+�A{��p��������C���A{��  �g
=��C��f                                    Bxb�q�  �          A+�A�����\������=qC���A�����(��S
=C�(�                                    Bxb��`  "          A*=qA�������&ff�r�\C���A����{�>{��  C��                                    Bxb��  �          A)A\)����=q�ģ�C�%A\)�\)��
=�33C�^�                                    Bxb���  "          A(��@�������)���lQ�C��@����z������33C�c�                                    Bxb��R  �          A+33A�����4z��tz�C�XRA��*�H��  ���C�                                      Bxb���  �          A+
=A������p��U��C�b�A���Z�H��
=�Ə\C��f                                    Bxb�ɞ  T          A)�A����\�����)�C�4{A��\(��z=q��33C�˅                                    Bxb��D  
Z          A*�\A��Q��n{��Q�C�9�A���G���G����
C�)                                    Bxb���  �          A)�A�
���H�ff�8(�C�<)A�
�g
=�����{C��                                    Bxb���  �          A*{A(����R�
�H�<��C���A(��O\)�������C�Z�                                    Bxb�6  
�          A(��A	G���=q�
=�P��C��=A	G���{�H������C�f                                    Bxb��  "          A*ffAQ�����?�\)@��HC�Q�AQ���p���=q�	�C���                                    Bxb�!�            A((�@�������@^�RA�  C��@�����{>�
=@33C��                                    Bxb�0(  	�          A$Q�@߮��(�@^�RA��HC�>�@߮��>��?Tz�C���                                    Bxb�>�  �          A'�@������H@FffA��\C�.@����ᙚ�u����C���                                    Bxb�Mt  �          A)p�@�p���(�?�33@��HC��H@�p����ÿ�����\C���                                    Bxb�\  |          A)G�A Q����þ�ff�(�C���A Q����
�[���(�C��f                                    Bxb�j�  T          A)G�@�p���p��B�\����C�~�@�p�����\)��=qC���                                    Bxb�yf  �          A*�HAQ��\��(��C�AHAQ���p���  ��
=C��H                                    Bxb��  
�          A&{@�33�������ԏ\C�T{@�33������z��=z�C���                                    Bxb���  
�          A&�R@�ff���ÿ���RC��
@�ff��  ��33��\C�
                                    Bxb��X  
�          A&{@����\��=q�ffC�}q@����(����[{C���                                    Bxb���  
�          A$��@����=q�0���}�C��\@����(���  �(�C�^�                                    Bxb�¤  
�          A&=q@�����  �#33�e�C���@��������
=�{C���                                    Bxb��J  
P          A)��A ����Q��aG���33C�A ���;���(�� �C��)                                    Bxb���            A*ffA z���(�?޸RA�C���A z��Ϯ���\��Q�C��f                                    Bxb��            A*ffA���G�?�(�A*�HC��)A����
�z��I��C���                                    Bxb��<  �          A,��A ���Tz�?�G�A{C�޸A ���h�ý�G����C�
=                                    Bxb��  �          A0  A,z���
?
=q@4z�C�P�A,z��=q�����˅C�,�                                    Bxb��  
�          A0��A+����>�
=@��C��A+���ÿ5�j�HC���                                    Bxb�).  �          A1�A!�<��@HQ�A�Q�C��qA!����?�  @��C�"�                                    Bxb�7�  �          A1�A"{�\)@r�\A��RC��HA"{�l(�@=qAI�C���                                    Bxb�Fz  T          A1�A%G���{@333Ap��C�=qA%G��p�?�z�A#33C�N                                    Bxb�U   T          A4(�A*�R?p���.�R�e�@���A*�R@33��\�*=qA-��                                    Bxb�c�  
�          A2�HA0�Ϳ�����\�%C�AHA0�Ϳ�=q������
C�4{                                    Bxb�rl  "          A2�\A*=q�;�?Tz�@�=qC�L�A*=q�@  �
=q�333C�!H                                    Bxb��  �          A3\)A$  �@��@2�\Ai�C��\A$  �{�?�
=@�(�C��H                                    Bxb���  �          A5p�A)����@>{At(�C�� A)��`��?˅AC���                                    Bxb��^  �          A5��A.ff���?��
@�=qC�q�A.ff��R>W
=?���C��{                                    Bxb��  T          A733A4Q��
=��=q��{C�%A4Q��{�����
=C���                                    Bxb���  "          A5p�A)p��p  ?��@1�C�=qA)p��g
=��z���z�C��{                                    Bxb��P  �          A4  A (���(��\(����C��qA (���Q��:=q�qC��                                    Bxb���  �          A3�A$�������������HC�H�A$���H���2�\�h(�C��f                                    Bxb��  �          A4  A=q������\�(  C��A=q�Z=q�z�H��ffC�}q                                    Bxb��B  T          A333Ap����H�*�H�]G�C�~�Ap��=p������C��                                     Bxb��  T          A2�\Aff��p��U����C�0�Aff�>{��z���z�C�:�                                    Bxb��  �          A1A�R�c33�p  ���C��A�R�Ǯ����=qC�N                                    Bxb�"4  @          A+33@���?\(�@�B*�R@�G�@����Q�@�z�B C���                                    Bxb�0�  
�          A0  A�H?(��@�Q�B�H@�  A�H�Q�@�BffC�K�                                    Bxb�?�  
�          A2�RA"�H��p�@�Q�A�G�C��3A"�H�33@y��A�{C���                                    Bxb�N&  �          A2�\A'
=��=q@c�
A�
=C��{A'
=�>{@!G�AQ�C��                                    Bxb�\�  
Z          A1��Aff�Ǯ@��
A��C��Aff�Z=q@^�RA�33C�~�                                    Bxb�kr  "          A3\)A ���(��@���A��C��A ����(�@\)ANffC��
                                    Bxb�z  T          A1�A   �P  @c�
A�p�C��A   ��
=?�ffA�C��
                                    Bxb���  
�          A+�
A  ��(�@\)AK�C��\A  ��{>k�?��C�>�                                    Bxb��d  �          A&�\@�p�������p��!��C�5�@�p���p�����X��C�\)                                    Bxb��
  "          A'�@��
���
�����Ip�C�3@��
���(��u��C��                                    Bxb���  J          A,Q�@�G��xQ���(��(�C�ff@�G��!G�����5\)C���                                    Bxb��V            A*=q@�
=���
�(��c�C�L�@�
=��{��ff���C��                                     Bxb���  
�          A)G�@��
�(����R�RC���@��
����ff��
C�.                                    Bxb��  �          A1��A  �L(��y����Q�C���A  ������z��ӮC�}q                                    Bxb��H            A)G�@�G���Q�?@  @�=qC���@�G���
=� ���p  C���                                    Bxb���  �          A%��z��{@L(�A�(�C�G��z��!���G���z�C�Y�                                    Bxb��  T          A$��@*�H�z�@8��A�p�C�@*�H�녿�\)��=qC��                                     Bxb�:  �          A&�R@@����Q�@���BD�
C�J=@@���Q�@�ffA�p�C�w
                                    Bxb�)�  �          A%G�@��H�\@���B/G�C���@��H�(�@p  A�G�C���                                    Bxb�8�  �          A"�R@�=q��
=�7���ffC���@�=q��  �����33C��                                    Bxb�G,  �          A*=q@�Q���\)���\��z�C�P�@�Q������\�.p�C��                                    Bxb�U�  �          A*ff@��
�������
�HC�
=@��
����
=�[�HC��                                    Bxb�dx  �          A)@�Q�� Q���Q����HC���@�Q����
��ff�H33C��H                                    Bxb�s  T          A)G�@�Q�� ����=q���C���@�Q���G����+�
C��H                                    Bxb���  s          A*ff@XQ��Ǯ�����<�RC�:�@XQ��Q��G�L�C�h�                                    Bxb��j  g          A,(�?����p��=q�w�C��?�����)p�§�\C�1�                                    Bxb��  T          A+�
@AG���z���
�Z��C��H@AG���z��#33��C�s3                                    Bxb���  
Z          A+�
@_\)�������D�
C��q@_\)��\��\�\C��R                                    Bxb��\  
�          A+\)@e���ff��
=�G�C�w
@e���\)���k  C�!H                                    Bxb��  �          A*{@E���G�����-��C��\@E��S33��� C���                                    Bxb�٨  �          A*�H@w
=����\)�ffC�f@w
=�n{��\�nC��                                    Bxb��N  
�          A+
=@j=q��z���
=�#�C��\@j=q�aG��p��u��C��                                    Bxb���  �          A,z�@�������~{����C���@�����ff��p��!�
C��=                                    Bxb��  �          A/�
@У����c�
��z�C�^�@У���G�������
C�h�                                    Bxb�@  
�          A/\)@ٙ���
��G��33C�W
@ٙ�����~{��G�C��                                    Bxb�"�  �          A1�@�p���?��HA$��C��\@�p��=q��\)��C���                                    Bxb�1�  �          A2{@�����Y�����C�xR@���
=�������C���                                    Bxb�@2  �          A-�@�33�߮�Ǯ�z�C�W
@�33�n{�	���O��C�.                                    Bxb�N�  �          A.�\?fff�H���!p�(�C��R?fff?����'�¢�BdQ�                                    Bxb�]~  �          A.�H@A����\����a  C�b�@A녿fff�"�Rz�C��                                    Bxb�l$  �          A2�\@����=q@�(�AиRC�ff@���\)?�(�@��
C��
                                    Bxb�z�  |          A6�HA
=��
=@^{A�=qC���A
=��\>�ff@G�C��                                    Bxb��p            A)p�@�����\)@�p�BC��C��@������\@��A�
=C��{                                    Bxb��  �          A)p�@����=qA33BK�
C��H@����33@���Bp�C�Ff                                    Bxb���  �          A+33@��
���Az�B^  C�1�@��
��{@ǮB��C�5�                                    Bxb��b  T          A+33@�������@�33B>�RC��@����G�@�{Aݙ�C�\)                                    Bxb��  T          A*�R@���@���B)
=C���@���@�G�A���C��q                                    Bxb�Ү  �          A,��@������@�p�B&�RC���@������@��HA�Q�C�˅                                    Bxb��T  �          A)�@��
���R@��B>Q�C��f@��
��\)@�{A���C���                                    Bxb���  �          A+\)@��R��Q�@�z�B;ffC�` @��R� ��@�Q�Aܣ�C�s3                                    Bxb���  T          A)@��
��G�@�p�Aď\C�� @��
��?�{@�  C�ff                                    Bxb�F  �          A,  @����Q�@\)A���C��3@���(�?#�
@Z=qC���                                    Bxb��  "          A/\)@�
=���H@�{A�  C��H@�
=�
=?c�
@�p�C��3                                    Bxb�*�  T          A0(�@���ff@��A�z�C�S3@��?��@�G�C�q                                    Bxb�98  T          A0z�@�=q��Q�@��A�  C��@�=q��{?��@�ffC��{                                    Bxb�G�  
(          A0z�@��
��z�@~�RA���C��)@��
��?
=@EC�4{                                    Bxb�V�  "          A/
=@����Q�@K�A��C�3@���{��z��G�C���                                    Bxb�e*  �          A.{@�33��Q�@J�HA�(�C���@�33�{��z���C��                                    Bxb�s�  
�          A.�H@���p�?�A#
=C�)@�� (����R��z�C���                                    Bxb��v  �          A/\)@ٙ��=q@�RAR=qC���@ٙ��
=��������C�k�                                    Bxb��  T          A/�
@����\)@;�Ax��C�\)@��� z�\���RC�Z�                                    Bxb���  "          A/\)@���G�@{A;
=C�� @���zῢ�\��{C�&f                                    Bxb��h  "          A.�H@�
=� ��@�RAQ��C�xR@�
=�����G���=qC��                                    Bxb��  
�          A0z�@����Q�?�(�A$Q�C�/\@������������RC�
=                                    Bxb�˴  �          A/\)@��H�p�?�Q�A#
=C���@��H��\��\)��C��3                                    Bxb��Z  
Z          A/
=@Ӆ�	?��@�G�C���@Ӆ��R�G��?�C��                                    Bxb��   
�          A,��@�����p�@L(�A�p�C���@�������Q�   C��                                    Bxb���  |          A,��@ᙚ��
=?333@p��C���@ᙚ��=q�%�`��C�z�                                    Bxb�L  T          A,��@�z��׮@b�\A�\)C�H�@�z���33?�\@/\)C��3                                    Bxb��  T          A,��@�����(�@�A���C��@�����Q�?�33@�(�C��{                                    Bxb�#�  �          A*ff@�(����@�{A�33C���@�(��?��
@�\)C��                                     Bxb�2>  
�          A'�@$z��ff���R��33C��{@$z�����G��9��C��H                                    Bxb�@�  T          A��@8�����=p����C���@8����
=��=q�"ffC�                                    Bxb�O�  
�          AG�@�����׿�(��MG�C�L�@������
��
=� �
C�1�                                    Bxb�^0  T          A@�ff��{��ff�	G�C���@�ff��=q��=q�߅C��H                                    Bxb�l�  "          Aff@�  ��\)@
=qAap�C��=@�  ��Q�Q���=qC�C�                                    Bxb�{|  "          A�H@�
=���@AT(�C��@�
=���Ϳs33��\)C��)                                    Bxb��"  �          A�H@�
=��p���(��z�C�Ф@�
=��{�����33C��                                    Bxb���  
�          A�@�����      ���
C��f@�����(��?\)����C��                                    Bxb��n  �          A\)@����
=?�  @���C��@�����
��p��-G�C���                                    Bxb��  t          A{@�(����H?�33@�Q�C�5�@�(���
=��G��2{C�u�                                    Bxb�ĺ            AQ�@����ڏ\@%�A�ffC�� @�����G���{�(�C���                                    Bxb��`            A�@�����=q?�33AC��H@������ÿ�{�'�C���                                    Bxb��  4          A=q@�G��������C���@�G���������=qC���                                    Bxb��  T          A�@��\���;�=q���
C��{@��\�θR�E���(�C�R                                    Bxb��R  �          A��@����G��s33��  C���@���љ��u���C�o\                                    Bxb��  �          AQ�@�����  ��Q��G
=C��{@�����p���G���z�C���                                    Bxb��  "          A
=@�(��Ǯ�Vff��33C�b�@�(���p���G��  C�O\                                    Bxb�+D  T          A��@�G���ÿ�33� Q�C���@�G��߮��G���{C��H                                    Bxb�9�  "          A��@�����Q��P����p�C�p�@��������G��{C��                                    Bxb�H�  T          A�\@��H���*=q��
C��=@��H�У���p����C��3                                    Bxb�W6  T          A��@x���p��0������C��R@x�����
��ff��\C��
                                    Bxb�e�  "          Ap�@|(��������H�˅C��q@|(���ff��Q��4��C��\                                    Bxb�t�  
�          A��@���ۅ��=q���HC�@����\)��p��,�C�C�                                    Bxb��(  �          A@�{��
=�s�
���C�  @�{���R���
��C�33                                    Bxb���  �          A33@���������H����C�� @����l����\�=
=C��                                    Bxb��t  "          A  @�ff��������
=C�8R@�ff�AG�����U�\C�Q�                                    Bxb��  
�          A�
@�
=�����ff�.�
C���@�
=����z��j�C���                                    Bxb���  �          A�@�z���
=��(��0{C��\@�z���R���np�C��=                                    Bxb��f  �          A�\@c�
���������Y��C��@c�
�n{���RC��f                                    Bxb��  T          A  @AG��G���aHC���@AG�?����	���)A�ff                                    Bxb��  T          A�R@1���{��\)�&ffC��=@1��c33���u33C�
=                                    Bxb��X            A@.{���
��Q��E(�C��@.{����� C�K�                                    Bxb��  f          A@�R���R��Q��M��C�7
@�R�
=q�z�  C�~�                                    Bxb��  "          Ap�?޸R������j��C��
?޸R��{���Q�C�                                    Bxb�$J  �          A?�G��|������y�C�H?�G������R�=C�s3                                    Bxb�2�  
�          A�@z����H�����h\)C�{@z�}p��\)�=C�t{                                    Bxb�A�  
�          A�@"�\��G������H��C���@"�\�p��Q��HC�}q                                    Bxb�P<  T          A�H@ ���ҏ\��
=�G�C�o\@ ���z=q��G��l�
C�Y�                                    Bxb�^�  �          A=q@XQ���  �0����Q�C��@XQ�����
=�C��R                                    Bxb�m�  �          A33@j=q��(���=q�G�C�p�@j=q���R��R�Q�C�}q                                    Bxb�|.  �          Aff@p������������C�:�@p����������D{C��=                                    Bxb���  �          A�@q����W
=��z�C�e@q������{�$ffC��q                                    Bxb��z  �          A�@7����
�����RC�"�@7���z���
=�T�C�`                                     Bxb��   �          A�
@��\��ff��G�����C��
@��\��(�����3(�C��                                     Bxb���  "          A�
@�����H��������C�y�@����(���{�C�
C�q�                                    Bxb��l  T          A@�z���R��=q�&�HC�]q@�z����H��p���RC���                                    Bxb��  T          A  @W
=��G���Q���RC�%@W
=��ff��  �Az�C���                                    Bxb��  "          A
�\@K���R�����C��@K���G����  C�b�                                    Bxb��^  T          A	@���Å�h����33C�@����������+  C���                                    Bxb�   
�          A��@�p���  ����C�C�#�@�p���=q������\C���                                    Bxb��  T          A{@������?�33A@Q�C���@����
=��R��  C�g�                                    Bxb�P  �          Ap�@�Q���33@(�Au��C�]q@�Q����\>��
@{C���                                    Bxb�+�  �          Az�@�(����
?��RA#�
C���@�(���(������G�C�:�                                    Bxb�:�  �          A�@�ff�[�@��A�Q�C��@�ff��=q?u@�C��=                                    Bxb�IB  �          A�@�
=���H@�G�A���C�j=@�
=�333@qG�A�Q�C���                                    Bxb�W�  
�          A\)@��>\@���B�\@G
=@����Q�@���BQ�C�q                                    Bxb�f�  �          Az�@�ff�{@��
B"��C�}q@�ff��\)@�(�A�{C�`                                     Bxb�u4  �          Az�@��C33@�{B  C��q@����@l��A£�C�P�                                    Bxb���  
�          A�
@޸R�J�H@�ffA�C���@޸R���\@>�RA�\)C�Q�                                    Bxb���  �          A�@�����(�@8��A�p�C���@������
?�  @ҏ\C��\                                    Bxb��&  �          A�@�����\)>\)?p��C���@���������x(�C��)                                    Bxb���  T          A  @�G���{�.{��
=C�Ф@�G���\)�E���\)C�h�                                    Bxb��r  
�          Ap�@�����
=��=q�ᙚC��3@������
�\������C���                                    Bxb��  �          A��@��������5����C�˅@�����33������C�˅                                    Bxb�۾  �          AQ�@�p������J=q��=qC�H@�p����R��(���\C�Ff                                    Bxb��d  �          A	��@�  ���
�L����z�C��f@�  �g
=���H�
z�C�|)                                    Bxb��
  �          A
=@ȣ���ff�:�H���C�+�@ȣ��r�\���� ��C�n                                    Bxb��  T          AQ�@�����ff�aG���\)C��@�����G����H��
C��=                                    Bxb�V  �          A	p�@�  ��  ��G�� =qC�˅@�  �P����(��7
=C�q�                                    Bxb�$�  �          A
=@��\��{�QG���
=C�%@��\��z���33��
C��{                                    Bxb�3�  �          A�
@mp���녿���޸RC��R@mp�����r�\��Q�C�"�                                    Bxb�BH  "          A��@��\��G���
=�E�C��H@��\��33��  ��G�C��f                                    Bxb�P�  
�          Aff@�\)�������eG�C�  @�\)��Q���
=��p�C���                                    Bxb�_�  "          AQ�@����  �
=q�]C��H@��������p���\)C���                                    Bxb�n:  "          A
=@����\)�G
=��G�C��R@������������C�5�                                    Bxb�|�  �          Aff@�Q��ҏ\�(�����C�O\@�Q����R����  C��H                                    Bxb���  T          A(�@�����5���RC��3@���  ��33��C�j=                                    Bxb��,  �          A
�R@����   �_�C��@����
�s33��{C���                                    Bxb���  �          A
�H@�\)�y��<#�
=�\)C���@�\)�i����\)�C�Y�                                    Bxb��x  �          A�
A���P��=�Q�?��C��A���E�������Q�C��{                                    Bxb��  �          A�@�\)�n�R=L��>�33C�w
@�\)�`�׿��\�(�C��                                    Bxb���  
�          AQ�A  �Y���L������C��\A  �9�������H��C�S3                                    Bxb��j  �          Az�@�(��~�R�����HC���@�(��P���'�����C���                                    Bxb��  �          A�R@߮��p���{�A�C��@߮��{�c33��Q�C���                                    Bxb� �  �          A�\@���(���Z=qC�@��u�j�H��G�C��                                    Bxb�\  �          A
=@�G�����(Q���p�C�ff@�G��N{��Q��ծC�
                                    Bxb�  T          A�@�
=��(��:=q��(�C�]q@�
=�E��  �㙚C�e                                    Bxb�,�  �          A�@�G���p���=q�+�HC���@�G�����
=�Y  C��H                                    Bxb�;N  �          Az�@�z�������Q��6=qC�9�@�z�����{�h\)C���                                    Bxb�I�  �          A��@��������\)�#33C�*=@����1G���(��Yz�C��                                     Bxb�X�  �          A(�@�����  �����RC���@����^�R�θR�=�HC�                                    Bxb�g@  �          A�R@��������fff��C�޸@����^�R����(�C���                                    Bxb�u�  �          A ��@�\)�Tz�?�{AXQ�C�J=@�\)�p��?��@��C��q                                    Bxb���  T          A��@�
=�;�?�(�AG\)C���@�
=�U?(�@���C���                                    Bxb��2  �          A�R@�Q��\)@y��A噚C���@�Q��\(�@9��A�33C���                                    Bxb���  �          A=q@��R���R@�  B#(�C�!H@��R�=p�@�Q�B�\C�˅                                    Bxb��~  T          A�
@�녿��@L(�A�33C���@���7
=@
=A���C�O\                                    Bxb��$  T          A��@�\)�)��@mp�AӅC���@�\)�p  @%A��RC�޸                                    Bxb���  �          AG�@����&ff@K�A���C�  @����a�@
=Al��C��q                                    Bxb��p  �          A@�33���@��A�(�C�Ff@�33�P��@]p�A�{C�G�                                    Bxb��  �          A�@�=q�33@z=qA��
C���@�=q�_\)@9��A��C�u�                                    Bxb���  �          A�@�{�\)@���B (�C�g�@�{�c33@P��A�G�C���                                    Bxb�b  T          A@ᙚ�2�\@3�
A���C�0�@ᙚ�e�?ٙ�AB�\C���                                    Bxb�  "          Az�@�Q��|(�@h��A�C���@�Q���@
=Al��C���                                    Bxb�%�  �          A(�@�ff���@{Ay�C�9�@�ff���H?
=q@q�C��                                     Bxb�4T  "          A��@�(���\)?aG�@��C�9�@�(���Q�B�\���C�*=                                    Bxb�B�  T          A��@����=��
?
=qC�5�@�����33���C��H                                    Bxb�Q�  
�          A�R@�  ����@�ffA�33C��{@�  �G�@VffA�
=C���                                    Bxb�`F  �          A�@�\)����@�  B/  C���@�\)�p�@���B�RC�s3                                    Bxb�n�  T          A33?�  @�z�@�Q�BeG�B��?�  ?�=q@��B�{B(
=                                    Bxb�}�  �          A��>��@��R@���BdffB�
=>��?�@��B��
B���                                    Bxb��8  
�          AG�@   ?�p�@��B���B
��@   �E�@�B�=qC�ff                                    Bxb���  
�          A�@�Q�@�@��
BVz�A�33@�Q�=u@���Bk33?L��                                    Bxb���  �          @�\)@ff@O\)@ָRBl�HBX
=@ff?Y��@��B��RA�ff                                    Bxb��*  "          A z�?xQ�@o\)@�{Bs�RB��
?xQ�?�  @�G�B�BPz�                                    Bxb���  �          @�Q�?�\)@�33@�z�B/33B���?�\)@H��@У�BvG�Bz��                                    Bxb��v  T          @�p�@(��@)��@��Bo  B4(�@(��>���@�B�ǮA\)                                    Bxb��  "          @�33?s33@`��@У�Bs\)B���?s33?�Q�@��B�k�BM��                                    Bxb���  �          @��;��R@:=q@�B�(�B�L;��R?   @�  B�B�W
                                    Bxb�h  �          @����@XQ�@�G�Bx�B�=q��?}p�@��B��CǮ                                    Bxb�  "          @��@r�\���H?z�HAN�\C�O\@r�\�
�H>�z�@r�\C�{                                    Bxb��  T          @�\)?�p��\)��  �^ffC�/\?�p�����{��C��                                    Bxb�-Z  T          A�@Z�H���
��z��;��C�H�@Z�H����陚�q  C��                                    Bxb�<   �          A\)@�z���33��\)���C��=@�z��]p���{�8p�C��                                    Bxb�J�  T          A�@��H��(��>{����C�l�@��H�[�������33C�
                                    Bxb�YL  �          A  @����ff�7
=��
=C�.@���qG�������33C���                                    Bxb�g�  �          A33@������
=q�b{C�AH@���s33�c�
��C�f                                    Bxb�v�  �          A�\@׮�����,(����RC�O\@׮������{����C���                                    Bxb��>  �          A��@�\�n{�}p����C��@�\�Mp����f�HC��                                    Bxb���  �          A{@������?�\)A33C�!H@����\?.{@��C��                                    Bxb���            A\)@�  �
�H?@  @�G�C�/\@�  ��\<��
>\)C�                                    Bxb��0  
Z          A\)@�
=�+�=u>��C�n@�
=�#�
�J=q����C��3                                    Bxb���  �          A z�@�(��G��#�
��\)C��f@�(��<�Ϳ��\��\C�R                                    Bxb��|  �          A�@�(��B�\��(��W�C�#�@�(��G��4z���=qC���                                    Bxb��"  T          A�\@��
�J=q>�p�@/\)C��@��
�G��#�
��{C�,�                                    Bxb���  �          @�ff@׮�k�@ ��AjffC��@׮��(�?@  @�ffC�C�                                    Bxb��n  T          @�(�@ۅ�333?��Ad  C��H@ۅ�P  ?h��@�33C�P�                                    Bxb�	  
�          @�\)@�\)�'�?�(�A2ffC���@�\)�<��?z�@��
C��                                    Bxb��  
�          A (�@�  ���У��=C�
=@�      ��(��G�<��
                                    Bxb�&`  
�          @�{@���    ��\)�   <�@���>�(���ff��
@J=q                                    Bxb�5  
Z          @��H@�{�u����"�RC��@�{>B�\����#�?�Q�                                    Bxb�C�  "          @��@�33>�{?8Q�@�(�@%@�33=�G�?J=q@�p�?L��                                    Bxb�RR  "          @�
=@񙚿Tz�?u@�
=C�޸@񙚿�=q?(��@�ffC��=                                    Bxb�`�  �          @�p�@�G��(Q�?�ffA��C��@�G��9��>�(�@J=qC�%                                    Bxb�o�  �          @���@ڏ\�Y��?���A$  C��H@ڏ\�j=q>�=q@   C��f                                    Bxb�~D  T          @�z�@�33���?�
=Ab�\C�s3@�33��\)>�(�@FffC�aH                                    Bxb���  �          @�{@�  ����?У�AJ=qC�\@�  ���<��
=�C�Y�                                    Bxb���  �          A�@���x�ÿL�����C��\@���]p������k
=C��                                    Bxb��6  "          A
=@��H�[������z�C�ٚ@��H�"�\�Tz���z�C��H                                    Bxb���  �          Az�@�=q�g
=�g
=���
C�|)@�=q���G�����C��{                                    Bxb�ǂ  
�          @��@��R�+��I����G�C��@��R��{�tz��p�C���                                    Bxb��(  �          @�R@qG����@1G�A���C��f@qG����?h��@��C�o\                                    Bxb���  T          @�\)@����33=���?E�C��R@�����H���]�C�'�                                    Bxb��t  "          @�p�@������������C��R@��������|���=qC��{                                    Bxb�  �          @��\@��R���@�A�\)C���@��R���H?O\)@ƸRC�#�                                    Bxb��  
�          @�{@������
@:=qA�\)C�Ф@�����G�?�p�A?�C��f                                    Bxb�f  T          @�{�
=@�@ÅBv{CxR�
=>���@���B���C*aH                                    Bxb�.  T          @�  �a�@G�@���B]�CL��a�>.{@�\)Bq�C15�                                    Bxb�<�  �          @�33��>8Q�@�  BQz�C1�������
@���BG=qCG                                    Bxb�KX  T          @��
�E?aG�@��HB|ffC$(��E���
@�=qBz�RCF}q                                    Bxb�Y�  
�          @أ׿#�
��@�(�B�{CX���#�
���@��RB�8RC~!H                                    Bxb�h�  �          @�R��G�@Vff@�p�BG�C����G�?�z�@��RB:�C8R                                    Bxb�wJ  T          @����=q@&ff@W
=A��HC�q��=q?�  @~{B
{C$�                                    Bxb���  �          @�{�n{��p�@��\B[�CF@ �n{�3�
@�\)B;�CY�                                    Bxb���  
�          @�z��5���  @�Br�
CS�\�5��Z=q@��BEz�CfQ�                                    Bxb��<  �          @ۅ�l(���@��BE�\CU^��l(��u�@�\)B�
Cb                                    Bxb���  
�          @�=q��z�?!G�@�\)B@z�C,G���z�k�@�B>p�C?&f                                    Bxb���  T          @�����?
=q@���BP��C,�=���ÿ��@�
=BLffCBJ=                                    Bxb��.  �          @�
=���
��z�@׮B�u�ChO\���
�b�\@�\)BgG�Cz{                                    Bxb���  �          @���?����=q@�z�B�W
C�h�?���j�H@��\Bb�
C�L�                                    Bxb��z  �          @�=q��\�s33@���B�G�Cq���\�1�@�\)B�\)C���                                    Bxb��   
�          @�(�����fff@�p�B�\C\8R����:�H@ӅB��RCy�q                                    Bxb�	�  
�          @�ff@\(��J=q@���BG�C���@\(���33@�z�Bz�C�ff                                    Bxb�l  T          @�=q@�33��Q�@�G�B��C�u�@�33���R@\��A�33C��R                                    Bxb�'  
�          A��@=q�J=q@���Bh��C���@=q��33@��B0(�C�7
                                    Bxb�5�  T          A\)���Ϳ��HA\)B�#�C�� �������R@�=qBpffC�L�                                    Bxb�D^  T          A�R>u�*�H@�ffB��qC��>u���@��
B[�HC�k�                                    Bxb�S  �          A(�?@  �G�@��B���C���?@  ���R@��BNC�)                                    Bxb�a�  �          A�?�z��a�@���Bzz�C��H?�z���  @�\)B=33C�h�                                    Bxb�pP  �          Aff?����s33@�RBm\)C��
?�����@��B1(�C�~�                                    Bxb�~�  �          A��@��}p�@�z�Ba�HC��{@���\)@�Q�B'{C���                                    Bxb���  �          A33@]p��H��@�ffB\z�C�� @]p���z�@��B+z�C���                                    Bxb��B  
�          A�@�Q��Z�H@��BHp�C�˅@�Q���G�@�33B�\C�C�                                    Bxb���  T          A=q@����Fff@ə�BCQ�C��)@�����\)@�{BQ�C��3                                    Bxb���  
�          A�R@���O\)@�  B9
=C���@������@��
Bz�C�c�                                    Bxb��4  T          A�H@�p��K�@��
B�C�XR@�p���G�@���A��C��f                                    Bxb���  �          A�
@�\)�X��@�z�B��C���@�\)��p�@�Q�A��
C�ff                                    Bxb��  
�          A(�@����0  @˅BB�C�W
@�������@��B�\C�                                    Bxb��&  
�          A{@�����R@�{B,�
C��@����|(�@�G�B�
C�:�                                    Bxb��  T          A@�p��=p�@�ffBA
=C�˅@�p�����@���BQ�C��H                                    Bxb�r  
�          A33@c�
��
=@���B>�\C�\@c�
���R@�p�B{C��{                                    Bxb�   
Z          Aff@j=q�s33@�ffBFffC���@j=q��33@�B�C�1�                                    Bxb�.�  
�          AG�@����G�@�=qBP\)C�1�@�����G�@�
=B$z�C��                                    Bxb�=d  T          Ap�@�ff���@��HB3Q�C���@�ff����@�Q�B��C��R                                    Bxb�L
  �          Ap�@o\)����@�=qB�\C�aH@o\)���@Z�HA��HC���                                    Bxb�Z�  T          A��@q���p�@�ffBQ�C�
@q����
@g�A�{C�T{                                    Bxb�iV  
�          Aff@xQ���@��\B=qC��H@xQ���33@\(�A���C�7
                                    Bxb�w�  �          A  @Vff����@���B!
=C�4{@Vff��  @l(�A�{C��H                                    Bxb���  "          A�@8Q���{@�B�HC�j=@8Q��ٙ�@N�RA�33C�w
                                    Bxb��H  
Z          A�@%���p�@�B�\C��@%���@#�
A�\)C��                                    Bxb���  
�          A�@W
=���H@��B\)C���@W
=���H@5A��C�{                                    Bxb���  
�          A�\@S�
��G�@�p�Bz�C�l�@S�
�ҏ\@A�A���C�Y�                                    Bxb��:  
�          A ��@z=q���@�ffBffC�
=@z=q��33@8Q�A���C��\                                    Bxb���  "          A ��@�{��p�@N�RA��C�L�@�{���?��AX��C�Z�                                    Bxb�ކ  �          @�\)@��R����@eA�G�C�޸@��R��G�@�Ar�HC���                                    Bxb��,  �          @�{@����@@  A�Q�C�Ф@������?���A;
=C�"�                                    Bxb���  
�          @���@�
=����?�Q�Ad��C���@�
=���>�
=@E�C���                                    Bxb�
x  �          @�(�@�
=���?�Q�Ah��C��=@�
=�Å>�Q�@*�HC��3                                    Bxb�  T          @�p�@�p���z�?�{A=��C��3@�p���33<��
>8Q�C�p�                                    Bxb�'�  T          @�\)@   � ��@�G�B��C�g�@   �l��@���B\�C�,�                                    Bxb�6j  �          @�33?�=q��@�  B���C��?�=q�l(�@���Bo��C��                                    Bxb�E  "          @�(�?�
=�&ff@�\B��fC�L�?�
=���\@�p�BQ��C��q                                    Bxb�S�  �          @�  ?�33�@  @�  BuffC�(�?�33��=q@�Q�B@G�C�Ff                                    Bxb�b\  
�          @��H@�z��(�@�BN��C��@�z��h��@�p�B,�C�\)                                    Bxb�q  �          @�33@:�H�~�R@�G�B;��C��@:�H��ff@�33B	�C��                                    Bxb��  
Z          @�ff@����\@��B�C��@�����@<(�A�{C�                                    Bxb��N  �          @�>��u@�B�� C��>��z�@�{B��C���                                    Bxb���  T          @��
@*�H���@ҏ\B��C��@*�H�Y��@��BWffC�3                                    Bxb���  �          @�33@AG����@�{Bx  C�3@AG��`��@��BQ\)C�^�                                    Bxb��@  �          @��@XQ���@��
Bg��C���@XQ��i��@�(�BB{C�b�                                    Bxb���  �          @���@]p���
=?��HA�{C�  @]p����?5@�33C�/\                                    Bxb�׌  T          @�\@l�����ÿ�33�2�\C�@l������7
=��33C��q                                    Bxb��2  
�          @�\@j=q����\(���33C�` @j=q��  ������C�AH                                    Bxb���  
�          @�{?�ff��G��h�����C��{?�ff����33�.�C��                                    Bxb�~  
�          @���?��љ��z=q��p�C��)?����
��p��.�
C�Q�                                    Bxb�$  
�          A�?�����=q���\�
�C���?�����\)��Q��A33C��                                    Bxb� �  �          A  ?Ǯ��\)��
=�/�
C���?Ǯ��(��޸R�fp�C�P�                                    Bxb�/p  
Z          A�?�{��{����6
=C�o\?�{�u�����k  C��{                                    Bxb�>  T          A��@K����H��p��6�\C��f@K��N�R���c33C�AH                                    Bxb�L�  
�          A=q?˅���أ��b(�C�ff?˅������H�C�Ǯ                                    Bxb�[b  "          @���L����{��
=�I�\C��\�L���U���  {C���                                    Bxb�j  �          @���ff���
����5��C�ÿ�ff�j=q��{�k�RCzaH                                    Bxb�x�  T          A{��\)�n�R����y�
C�o\��\)������{C��H                                    Bxb��T  �          AG�>��aG��\)§�{C�O\>�?�����H¤{B��\                                    Bxb���  "          A�H?@  ��G��=q©W
C��f?@  ?�{��p��{B�                                      Bxb���  �          @�ff=�\)��\)���
¯��C�S3=�\)?Ǯ��RL�B���                                    Bxb��F  
�          @�논#�
@333���Q�B�W
�#�
@��ȣ��[  B�8R                                    Bxb���  "          @��>���?�  ��\)B�B�k�>���@,����  �B�.                                    Bxb�В  
�          @���@   ��p����
�Dp�C�z�@   �-p���ff�q
=C�`                                     Bxb��8  T          @�=q@5��p��(����C���@5��=q��R����C�q                                    Bxb���  �          @��
@33������  ���RC�q�@33��z���=q�2{C���                                    Bxb���  �          @�{@A����H?�(�A1�C��{@A���  �B�\����C��R                                    Bxb�*  �          @�@E��33�z���33C�%@E�Å�w���  C�l�                                    Bxb��  T          A�@S33���6ff���HC��R@S33���������C�E                                    Bxb�(v  "          A�\@8���ᙚ�8Q�����C�'�@8�������R���C���                                    Bxb�7  T          @�
=@0����  ��G��  C�Z�@0����=q��=q�4Q�C��                                    Bxb�E�  
�          @�\)@E���
���H�/�C��@E�O\)�ȣ��Z  C��\                                    Bxb�Th  "          @�(�@=p���Q���\)�ffC�)@=p���Q���(��B�C�e                                    Bxb�c  
�          @�(�@%���ff��p���HC��\@%���
=����FG�C��H                                    Bxb�q�  "          @�@N�R��  ��\)�C�l�@N�R�qG���Q��D�C�K�                                    Bxb��Z  "          @�{@2�\��(�@���B	\)C�0�@2�\��
=@8Q�A���C���                                    Bxb��   
�          @�{?��R���\@���B+  C���?��R�Å@o\)A�C�
=                                    Bxb���  T          A Q�@�{����@�\An�HC��@�{���H?
=@��C�=q                                    Bxb��L  �          A z�@������?�p�A
=C�H�@�����;u��G�C��                                    Bxb���  T          @�  @�{��  ?k�@�z�C���@�{��녾��c�
C��                                     Bxb�ɘ  "          @�R@?\)����(��SQ�C�+�@?\)�=p����h{C�                                    Bxb��>  �          @�@�33�,(���z��G�C�z�@�33�У���(��.��C��)                                    Bxb���  �          @�@��\��z��\)�a�C�H�@��\��=q�Fff����C��                                    Bxb���  �          @��@�\)�������X��C�K�@�\)��z��<(���Q�C��                                     Bxb�0  �          @��@��
��33@33A��C��q@��
��\)?��
@��C��H                                    Bxb��  "          @��@����  @"�\A���C�)@����?��RA�C�
=                                    Bxb�!|  
�          A��@��\��(����R�-p�C�T{@��\��p��&ff���C��=                                    Bxb�0"  �          A�H@���������S
=C�j=@������
�<(�����C���                                    Bxb�>�  
�          A(�@�=q��  �-p���  C��R@�=q�q��i����(�C��                                    Bxb�Mn  T          AQ�@�
=�|����  ���C��@�
=�333��Q��'��C�t{                                    Bxb�\  
�          A��@ƸR�G���{�\)C��=@ƸR��
�����{C���                                    Bxb�j�  
�          A��@�������z=q���C��@��ÿ\���� Q�C���                                    Bxb�y`  
(          A(�@�p�������\)��ffC��@�p��fff�����  C��                                    Bxb��  "          A�\@����\)�W
=��=qC���@�����
���
���C�aH                                    Bxb���  �          A�R@�=q��
=�W���C��R@�=q�XQ����R����C�q�                                    Bxb��R  �          A�@��H�
=�mp���\)C���@��H���R���\���C�                                    Bxb���  "          Az�@�?&ff�p������@��
@�?����a���Q�A5p�                                    Bxb�  "          @�{@��=�Q��Y����(�?E�@��?Tz��S33��\)@�                                    Bxb��D  "          @��@��Ϳ��
�z���p�C��f@��Ϳ�������33C��
                                    Bxb���  �          A ��@��XQ�Ǯ�5C�` @��N{��\)��C��                                    Bxb��  "          @�
=@�zΎ�?�\)A
=C�f@�z��ff?L��@��
C�E                                    Bxb��6  "          A�\A   ?���?�\@dz�A�A   ?��R?J=q@��\AG�                                    Bxb��  �          @���@��?
=?�
=A
�\@��\@��>��
?��
A{@
=                                    Bxb��  �          @�{@�(��\(�@z�A��C���@�(����@A�z�C���                                    Bxb�)(  �          A�R@����#33@��
B��C�"�@����Z�H@n�RAٙ�C��                                    Bxb�7�  �          A  @�(��8Q�@��B	�C�h�@�(��s33@y��A�33C�q                                    Bxb�Ft  �          A�\@�  �G
=@�ffA�=qC���@�  �z�H@\(�A�(�C��\                                    Bxb�U  T          AG�@�G��  @�p�B�C�(�@�G��O\)@��HA�  C�aH                                    Bxb�c�  �          A\)@�G�����@��B2Q�C�h�@�G��1�@�ffB!=qC�+�                                    Bxb�rf  
�          A ��@�33�e@~�RA�RC��q@�33��33@H��A�C�@                                     Bxb��  "          AG�@�33��@u�A��C���@�33��(�@1G�A��C��R                                    Bxb���  
�          A
=q@�ff�%�@��B6p�C�XR@�ff�s�
@�\)B{C���                                    Bxb��X  
�          A
=q@�zΌ��@�  B)\)C�Z�@�z��'�@�=qBG�C�w
                                    Bxb���  T          A	�@��
�*=q@�(�B	\)C��@��
�g�@�
=A陚C���                                    Bxb���  �          A	@�G����@/\)A�z�C���@�G���G�?��ABffC�k�                                    Bxb��J  T          A	p�@Ӆ��\)@@  A���C��@Ӆ��  @ ��AZ=qC�q�                                    Bxb���  "          A	��@�ff�r�\?\A$��C���@�ff����?5@���C���                                    Bxb��  
�          AQ�@�G�����?@  @��C��@�G���33����}p�C��
                                    Bxb��<  |          A
=A��&ff@�A\(�C��A��>{?�  A=qC��                                    Bxb��  �          A�H@��
�e@�(�A�C�7
@��
���@c�
A�=qC��=                                    Bxb��  �          A��@�  ��  @��B(  C�XR@�  ���@��B�
C��f                                    Bxb�".  T          AG�@��\�%�@�B)��C�
@��\�hQ�@�G�BffC��                                    Bxb�0�  
�          A�
@c�
�R�\@�z�BWz�C���@c�
���@��\B4�C���                                    Bxb�?z  
�          AG�@���8��@�z�BB��C���@����=q@�B&ffC�#�                                    Bxb�N   
�          AG�@�  ��(�@�\)A�C��@�  ���
@@��A��
C��)                                    Bxb�\�  
|          A(�@�����
@dz�A�=qC���@�����@$z�A���C�Ф                                    Bxb�kl             A�R@����U�@��B  C�}q@�����@vffA�{C�t{                                    Bxb�z  �          A�
@�p���z�?h��@ӅC��@�p���{��G��J�HC��{                                    Bxb���  
Z          A�@?\)��
=@��B�C��{@?\)�Vff@���Ba�
C��                                    Bxb��^  "          Az�@g
=?xQ�@���B�ǮAq��@g
=�
=@��HB���C�S3                                    Bxb��  �          A(�@r�\�#�
@�G�B��C��@r�\�Ǯ@�z�Bw�C���                                    Bxb���  "          AQ�>B�\���@���BK{C�'�>B�\��(�@��\B�C���                                    Bxb��P  
�          A
�H@j=q���R@�ffBo�RC���@j=q�W�@��BU  C���                                    Bxb���  
�          A	@=p���(�@|(�A�33C�]q@=p���G�@-p�A�\)C�.                                    Bxb���  T          A	�@���\=�Q�?z�C�R@�� Q��G��$(�C�:�                                    Bxb��B  T          A	�@	���  �aG���  C�L�@	��� �Ϳ�=q�F=qC�y�                                    Bxb���  
�          A
�R?�{��G��Tz���
=C��{?�{��  �����p�C�xR                                    Bxb��  �          A  ?�����6ff��=qC�� ?��������G�C�AH                                    Bxb�4  �          A�
>�����������C�J=>�����p���Q��'z�C�}q                                    Bxb�)�  T          AQ�L������(���RC��ͽL����33��p��5G�C���                                    Bxb�8�  "          A����R��p�������C�����R������\�-�C���                                    Bxb�G&  T          A��>.{��\)�����0�\C��>.{��ff��(��\�HC��                                    Bxb�U�  �          A�?333�љ���p��"�
C��?333���H��=q�NC��)                                    Bxb�dr  �          A\)@<(���{��G���C���@<(������z��3{C��3                                    Bxb�s  "          A
=@{���(��x�����C�T{@{��������\��C�"�                                    Bxb���  �          A
=@�G�����G�����C��R@�G���Q����H��C��                                    Bxb��d  "          A
=@R�\�ۅ�s33��ffC���@R�\������G��(�C�Y�                                    Bxb��
  �          A
�\@�p���ff�2�\����C��@�p�����z=q��=qC�xR                                    Bxb���  �          A�@�����\)�#�
�uC�Z�@������
����z�C��q                                    Bxb��V  "          Az�@�  ���@���B(�C���@�  ���\@z�HA��HC�L�                                    Bxb���  
�          A\)@aG���(�@�Q�B33C��3@aG����
@Q�A�Q�C���                                    Bxb�٢  "          A��@������@��HB��C�G�@������@Tz�A��C�H�                                    Bxb��H  �          A=q@�Q����H@�z�A��C�B�@�Q�����@G�A�C�w
                                    Bxb���  
�          A�@W
=��?xQ�@��C��f@W
=������J�HC���                                    Bxb��  T          A�@5���\�S�
����C��@5����H����=qC��                                    Bxb�:  �          A  ��p����þ���Z�HC�����p���녿�z��^ffC�w
                                    Bxb�"�  
�          A��
=q���?�\@VffC���
=q��Ϳ�ff��ffCn                                    Bxb�1�  T          A  �s33� ���<����(�C����s33����(���z�C�O\                                    Bxb�@,  �          A(���p����(����
=C�����p���  ���H��C�}q                                    Bxb�N�  "          A(��?\)��33�G��r{Cy(��?\)�陚�j�H�Ǚ�Cw�q                                    Bxb�]x  �          A  ���	G����H���C�Ff�����0����{C�1�                                    Bxb�l  d          AG�>��(����[
=C��>�����e��p�C���                                    Bxb�z�  T          A(�<��
��p��e���C��<��
��z���
=�z�C�{                                    Bxb��j  
�          A
�R�������\�����\C������z����\�  C��                                    Bxb��  
�          A
ff��z��z��33�xQ�C�H��z���
=�p���υC��                                    Bxb���  
j          A	��R�33��=q�{C��׿�R�G��5���  C��=                                    Bxb��\  �          A	녿Q��������\C�@ �Q��=q�+���33C�!H                                    Bxb��            A
�\��33��\�:�H��(�C�B���33�ff��R�qG�C��                                    Bxb�Ҩ  
�          A	�����H�ff?J=q@�G�C�쿺�H��\�:�H��(�C�3                                    Bxb��N  h          A	G��.{��
@�Aa�C����.{��?�@w
=C��R                                    Bxb���  �          A�׾�(����@��A�=qC�xR��(���\?u@���C��f                                    Bxb���  T          Aff@�������33�
=C�!H@�����\�B�\����C�C�                                    Bxb�@  �          AG�@�  �K���{��
C��R@�  �
=q��p��#\)C���                                    Bxb��  �          A(�@���n{������C�:�@���+���(��+  C�{                                    Bxb�*�  "          AQ�@�Q���G��p  ���HC��{@�Q��������
� �HC�ٚ                                    Bxb�92  T          A=q@�(���p���\)��C��@�(���Q���=q�=qC��3                                    Bxb�G�  "          A�@��������{�癚C�/\@����ff��z����C�b�                                    Bxb�V~  �          A�@������������\)C�5�@���������ffC��q                                    Bxb�e$  T          A
=@������R��z���HC�l�@�����\)�����/
=C�=q                                    Bxb�s�  
�          A	�@�\)��Q������ �\C�3@�\)�~{��=q�=��C�ff                                    Bxb��p  
�          A
=q@�����  �����z�C�^�@����������$��C�                                    Bxb��  
�          AG�@�(���33�K����C�>�@�(���ff�������C���                                    Bxb���  T          A{@��R��p��L����(�C�j=@��R��Q���33��=qC���                                    Bxb��b  
�          A@�=q��  ���H�MC�!H@�=q����Fff����C�3                                    Bxb��  "          A�R@�{���ÿ޸R�4��C�7
@�{���
�;�����C��                                    Bxb�ˮ  "          A�@�ff��
=�\)����C�8R@�ff���R�[����
C���                                    Bxb��T  
(          A(�@�\)���R��33��{C���@�\)���
����C��=                                    Bxb���  �          A
=@��H��(���z���{C�� @��H��G���G���\C��                                    Bxb���  "          A�@�ff��G����H���
C��f@�ff��  �������C���                                    Bxb�F  D          A�@��
���\��{���HC���@��
������=q�

=C��\                                    Bxb��  
�          A(�@�Q���
=���R���C�XR@�Q��s�
��{��HC�Ff                                    Bxb�#�  �          A  @�����p���{�Q�C��@����L(���=q�*Q�C���                                    Bxb�28  "          A��@���������H�
ffC�n@����Tz���\)��C��3                                    Bxb�@�  T          A{@�
=�xQ������33C�}q@�
=�;���33�%z�C��=                                    Bxb�O�  �          A��@�z��n�R��{��C�Ф@�z��0������*��C�n                                    Bxb�^*  
�          A��@��
�^�R��
=�\)C�.@��
�#33����#z�C���                                    Bxb�l�  D          A��@�ff�.{��  �p�C�'�@�ff��G���(��)�\C�{                                    Bxb�{v  
�          A
�H@���
=q����\C��@�녿�����\)��C�33                                    Bxb��  �          A	p�@�
=�\(�����=qC��@�
=�%����  C���                                    Bxb���  
Z          A	G�@�G��]p���Q����C��@�G��%������ �C�o\                                    Bxb��h  
�          A
ff@�z��|(���z����C��@�z��B�\��
=�'{C��)                                    Bxb��  "          A
=@����  ���\�Q�C�!H@���fff��Q��&33C�4{                                    Bxb�Ĵ  "          A
�H@��H����z��"C�� @��H�\(���G��:��C�L�                                    Bxb��Z  T          A	�@��H��{��ff��RC��@��H�vff����!�C�u�                                    Bxb��   
�          A�@����{�(�����
C��\@������l����ffC�Ǯ                                    Bxb��  
Z          A	p�@�z���G��#�
��  C��H@�z���G��c�
�ŮC�Ф                                    Bxb��L  "          A
�R@�\)��33�G��s�
C�O\@�\)��z��W
=��p�C�C�                                    Bxb��  D          A	p�@U������	�C�5�@U�����Q��*=qC��                                    Bxb��             A	G�@I��������\)�G�C�P�@I����G���33�?�\C��                                    Bxb�+>  
�          A	G�@'���  ��\)�(�C�C�@'�������H�Jz�C�}q                                    Bxb�9�            A��@0  ������{�&��C��H@0  �����љ��G��C�H                                    Bxb�H�  
8          A	@1G���33��(��$�C���@1G������  �EQ�C��{                                    Bxb�W0  T          Az�@7
=��=q��z��)  C�3@7
=��G���Q��I��C�h�                                    Bxb�e�  D          A�
@?\)����33�)Q�C�� @?\)�����{�I�C�U�                                    Bxb�t|  
�          A(�@G
=���R����F�
C��R@G
=�e����H�dz�C���                                    Bxb��"  
�          A��?������H��Q��J�RC���?����������n(�C���                                    Bxb���  �          A��=�����{�F�\C���=���������k�\C��                                     Bxb��n  �          AQ�E�������H�M�\C��ÿE�������
�r
=C�޸                                    Bxb��  
(          A��������R���
�\G�C�{�����p����=q�C�w
                                    Bxb���  
�          A\)��\������{�`��C�����\�e���
�\C��q                                    Bxb��`  "          A  ��Q����������[�C�xR��Q��s�
��3333C�L�                                    Bxb��  
�          A
�R>�G����
����^�RC���>�G��j=q���H��C�l�                                    Bxb��  
�          A	>�����ff���H�R�
C�˅>���������\�wQ�C�O\                                    Bxb��R  
�          A	��?�G����������A��C�n?�G���{��ff�e33C���                                    Bxb��  T          A	��@ �����H�ٙ��Q�\C�@ @ ���l����\)�r�C�8R                                    Bxb��  
�          A��?�{��(������B��C�S3?�{������p��ez�C�B�                                    Bxb�$D  
J          A��?��������ff�C�C��\?����
=��R�eG�C��                                    Bxb�2�  
�          Az�?���\)��(��MG�C��?��w���\�n�RC��                                    Bxb�A�  T          Az�?�  ��������3G�C�l�?�  ��Q���=q�U��C�
                                    Bxb�P6  
Z          A33?�\)��{��{���C��?�\)������(��:�
C�&f                                    Bxb�^�  T          A�?���=q���H�^z�C�ff?��L(����~�C��\                                    Bxb�m�  T          A��?����������\\)C���?���I����\)�|ffC�g�                                    Bxb�|(  "          A	p�?��H���
��z��V=qC�^�?��H�n�R����x=qC���                                    Bxb���  
�          A(�?Q������߮�]
=C��?Q��hQ������C�b�                                    Bxb��t  
Z          A��?��R��{��ff�\p�C�n?��R�c33��33�~�C���                                    Bxb��  T          A�\?B�\���\��(��i\)C���?B�\�J�H��
=33C��)                                    Bxb���  �          AQ�@���ff��{�%(�C���@�������  �E�C�w
                                    Bxb��f  	�          A��@-p���p���Q���
=C�@-p���
=��  �G�C�5�                                    Bxb��  
Z          A33@1G���\)�H����C���@1G����������C��3                                    Bxb��  �          A (�@$z���=q��33�\��C���@$z���{�@����33C�&f                                    Bxb��X  T          @��R@,(���z��
=q�|��C�T{@,(���
=�N�R��z�C���                                    Bxb���  �          A Q�@Z=q�߮�˅�9�C�@Z=q����)�����C���                                    Bxb��  �          A (�@U���ÿ�ff�4��C���@U�ָR�'���  C�8R                                    Bxb�J  �          A (�@s33�ڏ\�ٙ��D  C��f@s33�Ϯ�.�R��p�C�(�                                    Bxb�+�  T          @�=q@P����33�a���(�C�3@P�������p��
��C�^�                                    Bxb�:�  
�          @�@�����z��<(�����C��@������
�qG���  C��                                    Bxb�I<  "          @�@��\��\)�:=q��C��R@��\���R�p  ��(�C�H                                    Bxb�W�  �          @�G�@w
=��(��I����{C�8R@w
=���\�~{��
=C��)                                    Bxb�f�  �          @��@X����G��Vff��ffC�33@X�����R��{�
=C���                                    Bxb�u.  
�          @�@8����(��:=q����C�33@8������vff���C�#�                                    Bxb���  �          @�{@n�R�����Y����  C�J=@n�R���R��(��z�C���                                    Bxb��z  �          @�@�33��33?\)@�(�C�)@�33���
��z���HC�\                                    Bxb��   !          @�R@������R@��\B\)C���@�������@W
=A��C�J=                                    Bxb���  
�          @�
=@�����ff@/\)A���C��R@������?��HAuG�C���                                    Bxb��l  
�          @�=q@�(�����@X��A�
=C�y�@�(���  @%�A��C�C�                                    Bxb��  
�          @�
=@XQ��`  ��p��"\)C�  @XQ��5������:{C�f                                    Bxb�۸  "          @�G�@U����dz����C��=@U����\���H�
=C�Q�                                    Bxb��^  "          @��R@�33��\)��\�w
=C�t{@�33�\��Q��/
=C�Ǯ                                    Bxb��  "          @�G�@�=q�ȣ�>��R@  C��@�=q��  �#�
���C��                                    Bxb��  
�          @�=q@���Ǯ?��A8Q�C�S3@������?
=@�(�C��)                                    Bxb�P  T          @��@�\)�dz��e��z�C��{@�\)�?\)���H���C��                                    Bxb�$�  "          @�\@�ff�`  �7����\C���@�ff�A��W
=��C��                                    Bxb�3�  T          @�G�@\���]p������:  C�xR@\���)������P��C�@                                     Bxb�BB  T          @���@�z����R>�?�G�C��f@�z�����0����  C�                                    Bxb�P�  "          @��@��Vff�����w�C�@��AG�����\)C���                                    Bxb�_�  T          @�G��u�Dz���ff�r\)C��u�{��33B�C��=                                    Bxb�n4  �          @�@<(���{���H�*�HC��@<(���{��\��{C�33                                    Bxb�|�  �          @�
=@!���=q>�Q�@;�C��f@!��љ���R���C��                                    Bxb���  �          @�{@   ����?�R@���C�8R@   �ٙ��Ǯ�I��C�33                                    Bxb��&  
�          @�R@$z���@ ��A�=qC�Ff@$z��Ϯ?ǮAIC�˅                                    Bxb���  �          @���@�33����   ��p�C�Y�@�33�����p��&�\C���                                    Bxb��r  �          @��@��H��Q�?+�@��C��@��H���������33C��                                    Bxb��  
�          @�=q@�(���Q쿂�\��{C��\@�(���G���33�f=qC�Q�                                    Bxb�Ծ  �          @��H@�p���\)��{��HC��@�p���  ��Q��g33C��q                                    Bxb��d  �          @�=q@�p����H�L�;�33C�W
@�p����ÿW
=��p�C��                                    Bxb��
  
�          @�{@�������.{��  C��@������ÿn{�ָRC�&f                                    Bxb� �  "          @���@��
���
�����\)C�q�@��
��G��h����C��=                                    Bxb�V  T          @��\@hQ�����?�(�A�ffC��@hQ�����?��A�C��                                    Bxb��  �          @��@�ff��z�@   A��HC�+�@�ff��{?���A8��C���                                    Bxb�,�  
�          @�{@�33���@8��A�
=C�O\@�33��
=@   Al(�C�~�                                    Bxb�;H  �          @��@�
=���
@EA�
=C��)@�
=��Q�@(�A�p�C���                                    Bxb�I�  
�          @�p�@��
��  ?��
@��
C��
@��
���H>.{?��
C���                                    Bxb�X�  �          @��@��R���׿�=q��C�
=@��R�������bffC���                                    Bxb�g:  �          @��R@��H������\)� Q�C�Y�@��H�����\)��ffC��)                                    Bxb�u�  T          A��@�33�����E���\)C�)@�33��Q��{���ffC�S3                                    Bxb���  �          Ap�@��
��=q��ff�(�C��\@��
����p��~=qC�c�                                    Bxb��,  �          A@�{��z῵�"�HC�%@�{�Å���\)C��H                                    Bxb���  T          A�@����p�>��@=p�C���@���������=qC���                                    Bxb��x  
�          AG�@��\�Ǯ>�G�@I��C��R@��\��\)����W�C���                                    Bxb��  T          A�@�������#�
��C��@�����녿���C�L�                                    Bxb���  T          A��@�z����þ8Q쿦ffC�y�@�z������p�C��=                                    Bxb��j  
�          A ��@�
=��G�>\)?xQ�C�"�@�
=��  �J=q���
C�=q                                    Bxb��  �          A ��@�����ff��p��(Q�C��@����ڏ\�����{C�G�                                    Bxb���  "          AG�@�Q���z�?u@���C�� @�Q���
==�Q�?�RC���                                    Bxb�\  �          A ��@�  ��Q�@��Az�RC�z�@�  ����?��A!G�C���                                    Bxb�  �          AG�@������\@ ��A�C���@�������?���AU��C���                                    Bxb�%�  
�          A Q�@��H�C�
@Q�Av�\C�T{@��H�U?�z�A?\)C�ff                                    Bxb�4N  �          @��@�=q��{?��A2�\C��R@�=q��33?:�H@���C�j=                                    Bxb�B�  �          A  @\)��=q��(��Q�C�,�@\)��=q����\)C��                                    Bxb�Q�  "          A\)@'���R�5��Q�C��q@'���
=�vff����C���                                    Bxb�`@  �          A
=?�(����K�����C��)?�(���ff��ff���C�0�                                    Bxb�n�  
K          A�H�������H����=qC�7
������=q���4(�C���                                    Bxb�}�  c          AQ��������������C|#׿�����\)����=ffCy�=                                    Bxb��2  �          A=q���
���H��
=�#(�C}�3���
��\)�ə��C  C{33                                    Bxb���  �          A�R�p��У���{��RCy^��p��������H�&�RCv�3                                    Bxb��~  �          A���z����|����G�C{xR�z�������p���Cy��                                    Bxb��$  �          A33�?\)��33�HQ����
Cvk��?\)��=q���H���Ct��                                    Bxb���  
�          A�R�����{�>{���HC{�=�����{�~�R��  Cz=q                                    Bxb��p  �          A Q�������0����  C}
=�����qG���C{�f                                    Bxb��  �          @�
=�s33�����3�
��C�K��s33��G��u���C��                                    Bxb��  �          @�{?L����R�K�����C�.?L����p���ff���
C�o\                                    Bxb�b  
�          @�
=?��H�����u���z�C�{?��H���������
=C���                                    Bxb�  "          @��>�
=����W
=���HC���>�
=���H�����RC��\                                    Bxb��  T          A ��?\)���
�n{��{C�AH?\)��Q���
=�G�C�xR                                    Bxb�-T  �          @��R?xQ�����������C�{?xQ���(���
=�C���                                    Bxb�;�  �          @��\>����R�<�����C��q>����ff�}p���=qC�H                                    Bxb�J�  �          @�G�����
=��\)��C�� ����G���(��&  C��)                                    Bxb�YF  �          @�p�@7���p���
=�((�C��@7���z��p����C�O\                                    Bxb�g�  T          @�z�@_\)�ᙚ�Ǯ�6ffC�'�@_\)��p���z��$��C�^�                                    Bxb�v�  T          @��R@�p���>�@W
=C���@�p�����\�n{C�H                                    Bxb��8  �          A{@����Q�?��A;�C���@����{?8Q�@��HC���                                    Bxb���  
�          Aff@xQ���\?u@��C�Z�@xQ����ͽ�\)��C�>�                                    Bxb���  �          A{@�\)��G�?��RA(��C��R@�\)��>��H@]p�C��3                                    Bxb��*  
�          A{@�z���=q?ǮA0��C��H@�z���\)?��@z=qC�Y�                                    Bxb���  T          Aff@�����  ?��AV�\C�h�@�����ff?k�@У�C���                                    Bxb��v  w          A�H@�=q��\)?J=q@���C��@�=q�أ׾L�Ϳ��C�                                      Bxb��  
i          Aff@���ٙ�?L��@�(�C�xR@����33�L�Ϳ���C�b�                                    Bxb���  
�          @��@_\)��p����c�
C��{@_\)��\��z��=qC��                                    Bxb��h  "          A ��@��H�ۅ?#�
@���C�k�@��H��(���p��'�C�b�                                    Bxb�	  "          A�@��\�޸R>�@P  C�/\@��\��ff�\)�}p�C�33                                    Bxb��  �          A@@  ���Ϳ��H�c33C�c�@@  �����>�R��
=C��                                    Bxb�&Z  "          A\)@a��޸R�(���
=C�n@a������[���Q�C�,�                                    Bxb�5   
�          A=q@]p���\)����
=C�+�@]p���=q�Tz�����C��                                     Bxb�C�  "          A�H@(���33�O\)���RC�y�@(��љ������C�33                                    Bxb�RL  
�          A�
@/\)��p��=p�����C�t{@/\)����~{��33C�,�                                    Bxb�`�  
�          A{@5���
�1G����RC���@5��(��s�
���HC�33                                    Bxb�o�  �          A\)@
=q���dz��Ǚ�C�*=@
=q��Q���33���C��
                                    Bxb�~>  
�          A33@\����p��0����33C���@\����{�q���33C��                                    Bxb���  "          Aff?��
���������z�C�R?��
��33��z����C���                                    Bxb���  �          A\)?��R��  ��p��33C�:�?��R��  ��33�&33C�1�                                    Bxb��0  "          A�
?�G�����\)�(��C���?�G������G��IG�C���                                    Bxb���  �          A�����׿�=q��{�=C_�����׼��
���¢aHC4�\                                    Bxb��|  �          @�\)>�����������,�C��=>��������=q�N=qC��                                    Bxb��"  �          A�@�\��z���  �	z�C��f@�\���
��p��)p�C��\                                    Bxb���  �          A�
@#�
�������R�C�R@#�
������\�/��C���                                    Bxb��n  T          A{@\(��أ��aG���33C�u�@\(�����\)�Q�C��=                                    Bxb�  
�          A��@�=q���H�G��aC�^�@�=q��
=�=p����C�%                                    Bxb��  �          A
�H@�
=���?�=qA�C�^�@�
=���>�@C�
C��                                    Bxb�`  �          A\)@�33�Å>k�?�ffC�z�@�33�\�(�����C��                                    Bxb�.  
�          A	�@�  ����?���ALQ�C��3@�  ��\)?��\@�  C�*=                                    Bxb�<�  
Z          A�
@љ���z�@C33A��C���@љ���G�@33At��C���                                    Bxb�KR  
�          A=q@�
=��@8��A�C�{@�
=����@
�HAb=qC��                                    Bxb�Y�  	�          Aff@�=q���@"�\A���C���@�=q��?�G�A733C��)                                    Bxb�h�  �          A��@��H���\@�Av�RC�"�@��H�Å?�Q�A�
C�xR                                    Bxb�wD  �          A�R@�  ���?��A�C�c�@�  ��>��?�Q�C�0�                                    Bxb���  T          A\)@����ff?��
Ap�C���@�����H>\@�C�e                                    Bxb���  
�          A{@E��
?(��@�=qC�G�@E�  �\)�g�C�Ff                                    Bxb��6  1          A�@��R����n{���
C���@��R��{� ���UC��                                    Bxb���  
�          A�@x����  �B�\��G�C�S3@x���񙚿�33�IC��                                     Bxb���            A	@l(����
�\�#33C��=@l(�����(Q�����C�b�                                    Bxb��(  
�          A	�@E����	���h��C��@E�����P����G�C�xR                                    Bxb���  �          A	@��
��33�k�����C�]q@��
��(��   �W33C���                                    Bxb��t  
�          A	�@�ff���H@^�RA�33C�K�@�ff���@5�A�(�C��                                     Bxb��  �          AQ�@�  �W
=@��RB (�C��@�  ���\@��
B�C��=                                    Bxb�	�  
�          A��@�\)����@��
Bp�C�>�@�\)��\)@�z�B  C�7
                                    Bxb�f  �          A�@��
�O\)@��B Q�C�]q@��
��{@�\)B��C��                                    Bxb�'  
�          A�H@��   @��HB�C��@��1�@�\)BQ�C��f                                    Bxb�5�  "          A{@ə����
@R�\A��
C�� @ə����@&ffA�p�C�O\                                    Bxb�DX  T          A@��H�l��@^�RA��
C�XR@��H��p�@8��A���C�Ф                                    Bxb�R�  
Z          A�R@�(���@Mp�A��C�� @�(����@!G�A��C�\)                                    Bxb�a�  
�          A�@������@\)A���C��{@����33?��HA:�\C�ٚ                                    Bxb�pJ  
i          Ap�@�p�����@0  A���C�@�p����@   A_
=C��                                    Bxb�~�  "          A�H@�����ff?�
=A[33C�O\@�����?���@��RC���                                    Bxb���  
�          A�H@�33���\@p�A��HC���@�33��z�?�A=��C���                                    Bxb��<  "          A
=@θR�{�@EA�  C�W
@θR��33@{A���C�f                                    Bxb���  c          A�@Ǯ����@I��A�(�C���@Ǯ��ff@ ��A�G�C�=q                                    Bxb���  
�          A (�@�33��Q��R��\)C�H@�33�����33�#�C�k�                                    Bxb��.  T          A�\@�  ���R�p������C�Ff@�  ���׿����5C���                                    Bxb���  
�          A�H@����#�
?�ff@��C��q@����,(�?&ff@��HC�O\                                    Bxb��z  
�          Az�@���(�?���AZffC�+�@���,��?ǮA.{C�E                                    Bxb��   
Z          A��@����{?\A(��C�Q�@������?�G�A�C��R                                    Bxb��  
�          AA�Ϳ#�
?(�@��C��=A�Ϳ:�H?   @\(�C��                                     Bxb�l  
Z          A33@�����?���AN�RC�Ф@����H?�{A3\)C��f                                    Bxb�   
�          A33@�z��?�33A�\C��q@�z��!�?��
@�\C�                                      Bxb�.�  �          Az�@�
=�>{=��
?z�C�}q@�
=�<(���p��#33C��\                                    Bxb�=^  T          A(�@��aG���\)��C�O\@��^{������RC�y�                                    Bxb�L  "          AG�@�G��^�R>Ǯ@-p�C��)@�G��`  �\)�xQ�C���                                    Bxb�Z�  �          Ap�@�z��vff�8Q쿠  C�8R@�z��q녿E�����C�q�                                    Bxb�iP  
�          Az�@����|(��#�
��\)C���@����w��B�\��G�C�                                      Bxb�w�  
          Aff@�G���(�?\A,��C��@�G�����?@  @��HC��q                                    Bxb���  T          A{@�\)��\)@�AyG�C�g�@�\)��Q�?�
=A"=qC��)                                    Bxb��B  
�          Aff@����{@ ��Ad��C��H@����{?�p�A
�RC��\                                    Bxb���  	�          A\)@�ff��ff?�(�A%C�� @�ff���?�R@��\C�T{                                    Bxb���  
(          A  @�p����H?�G�AQ�C�U�@�p���
=>\@)��C��                                    Bxb��4  �          A�H@������
?�z�A=qC�1�@������>��
@p�C��                                    Bxb���  "          A�R@�
=���\?�A�
C��@�
=��ff>��@8Q�C�L�                                    Bxb�ހ  T          Ap�@��H��G�?=p�@�Q�C�C�@��H���H���aG�C�#�                                    Bxb��&  "          A33@�G����R?Q�@�Q�C�Y�@�G�����=�?W
=C�"�                                    Bxb���  �          A�@�{���>��
@\)C��@�{�����Q��!G�C��                                    Bxb�
r  
Z          A�@�����>�(�@>{C���@�����{��=q��\)C���                                    Bxb�  T          A
=@�ff��p�>��H@[�C��=@�ff��{�u���HC�|)                                    Bxb�'�  �          A�R@��H�������
�.{C���@��H����E���C�'�                                    Bxb�6d  
�          A�H@�
=���R���H�Z=qC�B�@�
=���\��  �G�C��                                    Bxb�E
  T          A�\@�\)��z�333��p�C�y�@�\)��
=�����$z�C��{                                    Bxb�S�  "          A33@׮��z��ff�J=qC���@׮��Q쿕��C��                                    Bxb�bV  T          A\)@љ�������g�C���@љ����׿��
��
C��R                                    Bxb�p�  
�          A33@��
��\)�B�\��=qC�8R@��
�������R�(Q�C��)                                    Bxb��  
Z          A\)@�  ���H��ff��p�C��=@�  �����ff�L  C�Q�                                    Bxb��H  
�          A�@�  �������\)C�Y�@�  ��{����U��C�
                                    Bxb���  T          A�
@�����녿L�����\C��@�����(���(��%G�C��\                                    Bxb���  
�          A�@����(��ff���C�
@���mp��>�R����C�q�                                    Bxb��:  
�          A33@ָR�~�R�p���  C��f@ָR�b�\�C�
��C�{                                    Bxb���  T          Ap�@�
=���H����<z�C��@�
=���������=qC�                                      Bxb�׆  �          Ap�@�p���\)����0Q�C��\@�p���p���
��G�C�s3                                    Bxb��,  T          Ap�@�������\�K�C�G�@����z��#33���C�G�                                    Bxb���  
�          A ��@�\)��ff��ff��  C�Ǯ@�\)��
=�����R=qC�p�                                    Bxb�x  T          A z�@�33���R�S�
���C�33@�33�����33��=qC��                                    Bxb�  �          @��R@�
=����I����{C���@�
=��p��w����C�Y�                                    Bxb� �  �          @�
=@U�������HC�J=@U���
�����'{C�:�                                    Bxb�/j  T          @���@���{�|����33C�Y�@���
=��{�p�C�n                                    Bxb�>  "          @�33@�33��(��N�R��=qC�>�@�33��G������\)C��f                                    Bxb�L�  
�          @��@����Q��0����\)C���@����  �a���=qC�1�                                    Bxb�[\  "          @���@q�����`�����C�  @q���
=���
��
C��                                    Bxb�j  	�          @�33@�Q��z=q��  ����C�  @�Q��s�
�aG���\)C�K�                                    Bxb�x�  T          @��@���Fff=�\)?�C�t{@���Dz��(��HQ�C���                                    Bxb��N  
�          @���@�\�Z�H�k����C�q@�\�U�G���C�b�                                    Bxb���  T          @��H@陚�1G�?�@vffC��q@陚�4z�=�\)>��HC�s3                                    Bxb���  
�          @�=q@���QG�?\)@��C��3@���Tz�<��
>�C�k�                                    Bxb��@  �          @�  @�G��J=q?��@���C���@�G��L��<��
>.{C��                                    Bxb���  �          @�  @׮�q�>��R@G�C�Z�@׮�q녾��
�
=C�\)                                    Bxb�Ќ  �          @�\)@��H����<#�
=��
C�E@��H�~�R�&ff���HC�p�                                    Bxb��2  T          @�G�@��H���
=���?B�\C���@��H���H�z���Q�C��                                    Bxb���  �          @�\@���zῆff� (�C���@����Ϳ����aC�^�                                    Bxb��~  "          @��@�33��녿��
�\��C�k�@�33��{�.{����C�XR                                    Bxb�$  
�          @�Q�@:�H��z��K���z�C���@:�H���������\)C��                                    Bxb��  
Z          @�{@p����R�j=q����C�5�@p�������33�ffC���                                    Bxb�(p  T          @�
=?��H����=q�z�C�+�?��H��p���  �(��C�h�                                    Bxb�7  T          @�33?�ff�����ff���C���?�ff���R��33�0(�C���                                    Bxb�E�  �          @��?   ��p����� ��C��?   ��=q��ff�G33C�!H                                    Bxb�Tb  "          @�\)>��H��  ����Iz�C�y�>��H�HQ����R�o�C�n                                    Bxb�c  T          @�@l����33�X����ffC�Q�@l�����R����Q�C�3                                    Bxb�q�  T          @�\)@U��=q���R�{C�` @U��  ���R�3�C��                                    Bxb��T  �          @���@e������G��Q�C�aH@e��Q������,ffC��f                                    Bxb���  
�          @�Q�@�G���(��S33��G�C�k�@�G���  ��p��
=C�,�                                    Bxb���  �          @�p�@�=q��
=�0����33C��{@�=q��p��h����C�s3                                    Bxb��F  T          @��@��������R��z�C�(�@���fff�8Q���(�C���                                    Bxb���  T          @�ff@�33���R��{�$��C�=q@�33��������p�C�"�                                    Bxb�ɒ  
�          @�
=@|����  �O\)��  C�+�@|�����׿�{�g�
C���                                    Bxb��8  "          @�R@�\)��  �����  C���@�\)���R�����C�C�                                    Bxb���  T          @�@L����{��=q�(  C�4{@L�����
�=q��=qC���                                    Bxb���  �          @�Q�@S�
�љ���=q�A�C�h�@S�
���+���
=C�
                                    Bxb�*  �          @�@c�
�����p��T��C��f@c�
�����3�
��
=C�O\                                    Bxb��  �          @�p�@r�\��z���R����C�ٚ@r�\���
�`  ��Q�C�                                      Bxb�!v  �          @�33@tz���\)� ���v=qC���@tz���G��C33��=qC��
                                    Bxb�0  	           @�\)@2�\�ʏ\�-p���G�C��@2�\�����qG���C��f                                    Bxb�>�  
�          @�\�tz��]p���G��<33C^!H�tz��������T�\CT�)                                    Bxb�Mh  �          @�33���R��=q����\)Ca�q���R�]p�����/�C[n                                    Bxb�\  �          @���h���Vff����H=qC^���h���G������`�RCT                                      Bxb�j�  �          @���
=�@  ��ff�4p�CWn��
=�33�����I
=CM�f                                    Bxb�yZ  �          @�R���ÿ�{��\)�,�CF�����ÿB�\��{�6\)C=�                                    Bxb��   �          @�{���\�E���{�+��CWk����\�
=q�����@�\CN��                                    Bxb���  �          @�\)��33�z���
=���CI!H��33���H��Q��)z�C@��                                    Bxb��L  �          @�z�=�����33�i�����C��H=����x�������B(�C���                                    Bxb���  �          @�  ��  �xQ��dz��&C�{��  �L(���ff�M�HC|�H                                    Bxb�  �          @�z�@�Q��[����H��z�C���@�Q��R�\����7\)C�&f                                    Bxb��>  T          @�G�@�33�R�\?�(�A4��C��)@�33�`  ?^�R@ӅC�                                    Bxb���  �          @�\)@����K�?�z�A�RC�:�@����U�?�@��C���                                    Bxb��  �          @�
=@���E?��HA�C���@���P��?#�
@�{C���                                    Bxb��0  �          @�ff@����,(�?333@�p�C�XR@����1G�>W
=?�\)C��                                    Bxb��  �          @�  @�=q�
=?J=q@�Q�C��H@�=q�{>�33@*�HC�aH                                    Bxb�|  �          @�@�
=�Q�>�33@(Q�C���@�
=�	���#�
���
C��3                                    Bxb�)"  �          @�@�\)��R?��@�ffC��@�\)��
>B�\?�p�C���                                    Bxb�7�  �          @��@���(�?aG�@�C�'�@���#�
>�(�@UC��
                                    Bxb�Fn  �          @�\@�z��>�R?G�@\C��f@�z��E�>aG�?�  C��3                                    Bxb�U  �          @陚@����L��?�@�(�C��@����P  �u�   C�Ǯ                                    Bxb�c�  �          @�(�@Ϯ�^�R>��@�\C��@Ϯ�^{��p��:=qC��                                    Bxb�r`  �          @�@��
�U�<#�
=��
C���@��
�QG�������C��R                                    Bxb��  �          @��
@��H�o\)��\)��C���@��H�g��u��
=C�!H                                    Bxb���  
�          @θR@�  �s�
��  �9��C�� @�  �aG���
=��ffC���                                    Bxb��R  �          @�G�@��
�a녾�p��B�\C��@��
�Y�����\�(�C�w
                                    Bxb���  �          @�(�@����u�����j�HC�Q�@����k�����C��{                                    Bxb���  �          @��@��R�|(���Q��6�RC�H�@��R�fff�������C�n                                    Bxb��D  T          @�z�@�(��z=q��ff�k�C��q@�(��`���\)���RC��                                    Bxb���  �          @�=q@�  �HQ��%��z�C��3@�  �&ff�HQ���ffC���                                    Bxb��  �          @�{@�����
�j=q��  C���@���W�����G�C�~�                                    Bxb��6  T          @�@h���\(���p��={C�Q�@h������
=�WC�y�                                    Bxb��  �          @�@`  �`����ff�3�
C�u�@`  �!������O�C��                                    Bxb��  �          @���@G��G
=��Q��M��C���@G��G���  �h�C��                                    Bxb�"(  �          @��@�p���(���  �	Q�C��@�p��`  ����&{C��                                    Bxb�0�  T          @�@�\)����z��z�C��)@�\)�aG���z��,(�C��                                    Bxb�?t  �          @��@��\����?�G�@�33C�J=@��\����=�Q�?#�
C��)                                    Bxb�N  �          @�  @�p������
=�P��C�Y�@�p���{�����*�HC��
                                    Bxb�\�  �          @�{@�z���(��O\)�\C���@�z���(������[�C�7
                                    Bxb�kf  �          @�\)@�33��
=�����'�C���@�33���
����ffC��                                     Bxb�z  T          @�R@~�R��
=�8Q���C��@~�R���\�u��  C�}q                                    Bxb�  �          @�33@g������Q�����C�:�@g���33�QG���Q�C�޸                                    BxbX  T          @У�@Z=q��33��33�G�C�Ff@Z=q��\)������C�5�                                    Bxb¥�  �          @�33@Y����(����\�4Q�C�+�@Y����������=qC��                                    Bxb´�  �          @��
�z�H�HQ���{�f��C|�\�z�H����{Ct�f                                    Bxb��J  �          @޸R@�(����\��=q�2�\C�� @�(���
=������C��R                                    Bxb���  �          @�{@���l��@   A��C��)@�����
?�p�Az�RC�q                                    Bxb���  �          @��@�
=�w
=�
=q��ffC���@�
=�l(����
�)�C��R                                    Bxb��<  �          @޸R@����X���H����Q�C��@����.{�o\)�ffC�ٚ                                    Bxb���  T          @�ff@L����33�Q���{C�3@L�����\�E���
=C�y�                                    Bxb��  
�          @���?�������;���Q�C���?���������G��  C���                                    Bxb�.  �          @��
?�z���33�X����C���?�z�����������C�o\                                    Bxb�)�  �          @���@(���ff����ffC���@(���ff�Fff��=qC���                                    Bxb�8z  �          @��
@  ��p��ٙ��`��C���@  ���R�<(����C�S3                                    Bxb�G   �          @�@�z����׿�ff�j{C�o\@�z������6ff����C���                                    Bxb�U�  �          @�p�@�����������{C��f@�������������C�l�                                    Bxb�dl  �          @޸R@[����
?0��@���C�*=@[���z���H���C�                                      Bxb�s  �          @��@g�����>���@*=qC��@g����R�aG���p�C���                                    BxbÁ�  T          @��
@u���
=?.{@���C�Ф@u���\)�z���Q�C�˅                                    BxbÐ^  �          @��H@�����\?�33Az�C���@����ff<#�
=�Q�C�~�                                    Bxbß  �          @�Q�@��
��ff@ffA��RC��@��
����?�z�A�\C��                                    Bxbí�  �          @��
@�����R?!G�@�p�C��=@����\)��(��_\)C��)                                    BxbüP  �          @�p�@�{��  �z�����C��@�{��Q��(��_�C���                                    Bxb���  �          @�33@\)��(���=q��C��@\)��{��  �=��C��3                                    Bxb�ٜ  �          @��@Q���녿���ffC���@Q���=q����hz�C�1�                                    Bxb��B  �          @�
=@[���ff�fff��ffC�z�@[���z�������C�q                                    Bxb���  �          @�\@I����ff��Q���C�s3@I�����\����
=C�.                                    Bxb��  �          @ᙚ@���G��J=q��  C��{@��Ǯ�
=��Q�C�e                                    Bxb�4  �          @�(�?��H�����R�!�C�*=?��H�����%���p�C��                                    Bxb�"�  �          @�(�@���
=��G��e�C��H@���\)�C33��G�C��H                                    Bxb�1�  �          @޸R?�����ÿ��R�G
=C��f?���\�333���HC�.                                    Bxb�@&  �          @׮?��\�љ���33��C�h�?��\��p��{����C���                                    Bxb�N�  �          @��>.{���H����R{C�� >.{��(��7���z�C��                                    Bxb�]r  �          @�33=�����
=�	������C�t{=�����z��\(���\C��                                     Bxb�l  �          @�
=@���(������7�C�@���ff�+����C���                                    Bxb�z�  �          @��H@�
��z��R��
=C���@�
���
��Q���{C�\)                                    Bxbĉd  T          @�Q�@��\�����~�R���C�h�@��\?:�H�z�H���A
=q                                    BxbĘ
  �          @�=q@���E��K���RC��q@����p���{C�h�                                    BxbĦ�  �          @�@����n{�#�
���\C�"�@����Fff�Q�����C���                                    BxbĵV  �          @�{@�=q����������RC��
@�=q�}p��Fff��
=C��q                                    Bxb���  �          @���@��\��Q��!G���
=C��R@��\�xQ��Z=q��
=C��                                     Bxb�Ң  �          @ٙ�@�(�����33��z�C��3@�(���{�C�
��(�C�y�                                    Bxb��H  �          @��H@�G�������
=�c\)C�/\@�G����\�-p�����C��3                                    Bxb���  �          @��H@�=q��Q��G��Lz�C�޸@�=q����%����C��                                    Bxb���  �          @أ�@p�����ÿ�\�tQ�C���@p�������7
=��33C�)                                    Bxb�:  �          @�  @��\����\�P(�C�%@��\��
=�%��ffC�h�                                    Bxb��  �          @���@e�����ff�33C�K�@e��Q���R��ffC�(�                                    Bxb�*�  �          @�  @E��������U�C�O\@E����(��fffC��                                     Bxb�9,  �          @أ��
=q����?�AfffCz���
=q�˅>��R@'�C{@                                     Bxb�G�  T          @ٙ��^�R��z�>��
@.{C�B��^�R�љ���\)��\C�5�                                    Bxb�Vx  �          @�ff������
�W
=��C�� �����p�����e��C��
                                    Bxb�e  �          @�(��&ff��Q�>��H@���C�(��&ff�θR�p���33C�#�                                    Bxb�s�  �          @Ӆ��=q��33?E�@ڏ\C�0���=q�˅�!G����\C�33                                    Bxbłj  �          @�Q�#�
��z��\����C��{�#�
���
�����p�C���                                    Bxbő  �          @��
������?!G�@�(�C��3�����z�W
=��(�C��\                                    Bxbş�  �          @�33?�������>.{?��HC��?������ÿ���0��C��                                    BxbŮ\  �          @�(�?��׮?uAp�C�=q?���G������33C�9�                                    BxbŽ  �          @�Q�>.{��33?��A��C���>.{�������Z=qC��)                                    Bxb�˨  �          @θR������G�?�G�A5p�C�����������#�
��Q�C��                                    Bxb��N  �          @�  ��Q����?���AB�\C}�f��Q���Q켣�
�8Q�C~E                                    Bxb���  �          @��H��\��p�?fffA
=C{��\��
=��G��\)C{.                                    Bxb���  �          @��
�G���
=?L��@�C{L��G���������\)C{c�                                    Bxb�@  T          @�  �����p�?�\)A%p�Cs3������׾u��C�                                    Bxb��  �          @�
=��p���\)@'
=A̸RC~33��p���?�{AQG�Cff                                    Bxb�#�  �          @�(�������@*=qA���C�!H�����{?���AI��C���                                    Bxb�22  �          @�\)�#�
���
@$z�A�  C�箿#�
�ə�?�
=A(��C�q                                    Bxb�@�  �          @�
=�
=q�Ӆ���|(�C����
=q���H��Q����C��\                                    Bxb�O~  �          @ҏ\�����=q��Q��&�\C�=q�����z��%���C��{                                    Bxb�^$  �          @��Ϳ�  ���H�
=���C�𤿠  ���R�X���=qC�:�                                    Bxb�l�  �          @ə����H����>�G�@���C�T{���H����s33��C�Ff                                    Bxb�{p  �          @ʏ\������?���ABffCz�����ff�#�
��{C{�                                    BxbƊ  �          @ƸR>W
=���?�A.�\C��>W
=��p���  ��C��q                                    BxbƘ�  �          @�ff?
=q��33>���@n{C���?
=q���ÿ�G��Q�C��{                                    BxbƧb  �          @ə�?�����  >�{@EC�t{?�����p�����z�C��\                                    Bxbƶ  �          @�G�?(���(�?n{A	�C��
?(�����\��
=C�Ф                                    Bxb�Į  �          @ȣ�>��
��G�?\Aa�C���>��
��\)=��
?@  C�w
                                    Bxb��T  �          @љ�>\��z�@��A�33C�Ǯ>\��\)?5@�  C���                                    Bxb���  �          @���?���녾�(���33C��?���=q��33���
C�\)                                    Bxb��  T          @�
=@�����qG���C��@����p�����$��C��                                    Bxb��F  �          @Ӆ@�G��
�H���\�!\)C���@�G���33��
=�4�C�xR                                    Bxb��  �          @Ϯ@E��33��{���\C��
@E�����@  ��{C�(�                                    Bxb��  �          @�\)?������8Q���{C�o\?�����\�ff��p�C���                                    Bxb�+8  �          @�  ��z����H?�Q�Ah��C��쿴z����H>#�
?�33C���                                    Bxb�9�  �          @�z�>\)��G�?.{@�Q�C���>\)��\)��
=�
�RC���                                    Bxb�H�  �          @��?k�����>�\)@G�C�O\?k���������9��C�`                                     Bxb�W*  T          A   ?�\)��zῂ�\��(�C��?�\)���3�
��C�N                                    Bxb�e�  �          AG�@   ��Q��ff�0��C�4{@   ���R�\���RC��                                    Bxb�tv  �          A��?�
=��ff��  ��RC��?�
=��\)�3�
����C�w
                                    Bxbǃ  �          A Q�?�����33>k�?�33C���?�����p���z��?
=C���                                    BxbǑ�  �          A Q�?�\������33�   C�e?�\���p���Q�C���                                    BxbǠh  �          A z�?�(���Q쾨����C�"�?�(���ff�(��}�C�k�                                    Bxbǯ  �          A ��?����\>8Q�?�G�C��?���z��(��E��C���                                    Bxbǽ�  �          Ap�@X����\������C�ff@X����=q���\z�C�˅                                    Bxb��Z  �          A ��@X�����ü���  C�}q@X���ᙚ��ff�O33C��
                                    Bxb��   �          A   @!���?(��@���C�@ @!���\)��Q��	p�C�S3                                    Bxb��  �          @�
=@ff��(�?c�
@��C��3@ff����  ��ffC��
                                    Bxb��L  �          @�ff?ٙ����
?���A��C�K�?ٙ����R�
=��Q�C�9�                                    Bxb��  T          @��?�G���  ?
=@�G�C��f?�G������=q�=qC��
                                    Bxb��  �          A ��?h����p�?�\)A (�C�G�?h����ff�\(���(�C�C�                                    Bxb�$>  �          A   ?����?aG�@�=qC�:�?����\��=q��G�C�>�                                    Bxb�2�  �          @�ff?�ff��{?�
=A��C��?�ff����B�\��
=C��                                    Bxb�A�  �          @���?��
��Q�?��
A7
=C��?��
�����p��1�C��R                                    Bxb�P0  �          @�?�����?�z�A*�RC��?����׾��e�C��3                                    Bxb�^�  �          @�z�@ ����ff?��HAHQ�C���@ ����z�L�Ϳ�C�^�                                    Bxb�m|  �          @�?���G�?��A!�C��)?���z���z�HC���                                    Bxb�|"  �          @�p�?У����R�#�
����C��R?У���ff��(��g�C�+�                                    BxbȊ�  �          @��?�33��
=�z����RC�"�?�33��\�   ��  C�g�                                    Bxbșn  �          @�33?�����p�?W
=@�C�H?�����z῎{��HC�f                                    BxbȨ  �          @���?s33��G�?333@�(�C�}q?s33��
=���
�{C��f                                    Bxbȶ�  �          @�z�>����
=u>���C�w
>���(���Q��d  C�z�                                    Bxb��`  �          @��\�������?(�@�\)C�f�����p������#33C��                                    Bxb��  �          @�33�#�
��
=?�\)A z�C�h��#�
��녿#�
��ffC�k�                                    Bxb��  �          @�33?��R����?&ff@��C���?��R��녿����  C���                                    Bxb��R  �          @��H?��H��=q>Ǯ@7
=C�C�?��H��p��Ǯ�7\)C�g�                                    Bxb���  �          @���?�{���
�����AG�C��?�{��������C�o\                                    Bxb��  �          @��=u���
@��RB��C�@ =u��\)@�A��\C�8R                                    Bxb�D  �          @����(��ڏ\@uA�\C�5þ�(����H?��HAf�HC�c�                                    Bxb�+�  �          @���?�����?��@��\C�1�?������H����#�C�@                                     Bxb�:�  �          @�z�?z�H�񙚿�\�tz�C���?z�H����(����C���                                    Bxb�I6  T          @��?����񙚾��H�i��C��?�����p�����Q�C�9�                                    Bxb�W�  �          @�
=?������
��\�p��C��?�����\)�{��\)C�S3                                    Bxb�f�  �          @�(�?�G���Q�fff��=qC��R?�G���Q��8����ffC���                                    Bxb�u(  T          @�Q�?\(���p��=p���{C�1�?\(���R�-p���  C�e                                    BxbɃ�  �          @��?(�����L�����C�E?(���Q��333��  C�k�                                    Bxbɒt  �          @���?�  ��Q������C���?�  ��\�'����C���                                    Bxbɡ  T          @��?Q�����n{�ٙ�C�
=?Q���
=�;����
C�AH                                    Bxbɯ�  �          @�(�>��H���H��(��J�HC���>��H��{�\)����C��                                    Bxbɾf  �          @���\)���
�����UG�C�"���\)���R�S33��\C��H                                    Bxb��  �          @�33�Q���녿aG���C{Q��Q���33�!G���(�Cy�R                                    Bxb�۲  �          @�
=>�
=��Q�\�6�HC��)>�
=�ڏ\�]p��֣�C��                                    Bxb��X  �          @�{<��
��������"{C��<��
�����Z�H�̸RC�
                                    Bxb���  �          @��L����\)��(��H��C�B��L���߮�n{��ffC�.                                    Bxb��  �          @�(����������@Q�C�<)����޸R�h����=qC�                                    Bxb�J  �          @�33�+����Ϳ����8z�C�� �+���ff�dz���ffC�@                                     Bxb�$�  �          @�  ��{�޸R�=q���
C�y���{�����{�{C���                                    Bxb�3�  �          @���@�z����?��A�33C��R@�z���?��@�(�C��3                                    Bxb�B<  �          @�z�@�p��c�
@>�RA�
=C�@�p���\)?�33A���C��                                    Bxb�P�  T          @�  @�{��@���B+  C�xR@�{�N�R@s33B
�RC���                                    Bxb�_�  �          @�=q@����z�@?\)AЏ\C���@����G�?�
=Ad��C�
=                                    Bxb�n.  �          @�  @Z�H��Q�=#�
>�{C�^�@Z�H�������
�S\)C��
                                    Bxb�|�  �          @Ӆ�k������'
=��(�C�<)�k��hQ��p  �3�C                                    Bxbʋz  �          @�  ��ff�����aG���Q�C~W
��ff���H�����:�
Czc�                                    Bxbʚ   �          @���R����\����  Cv����R���H��Q��6�CqY�                                    Bxbʨ�  T          @��H�Fff�����:�H��{CnY��Fff������{��
Chs3                                    Bxbʷl  �          @��H�]p���{�'
=��G�Ck��]p������w��p�CeaH                                    Bxb��  �          @��>�R��=q�AG����\CnB��>�R�s33����%�Cgٚ                                    Bxb�Ը  �          @�=q�vff��=q�9���ԣ�CdO\�vff�Vff�~�R��
C]�                                    Bxb��^  �          @�=q�Z�H��\)������z�Cl��Z�H��\)�Tz���(�Ch�)                                    Bxb��  �          @��H�?\)�������A��Cr0��?\)��=q�9����  Cok�                                    Bxb� �  �          @Ӆ�A������s33�Q�Crp��A������!���=qCpG�                                    Bxb�P  T          @ҏ\�=p���p������:=qCrc��=p���=q�6ff��33Co��                                    Bxb��  �          @�p��{���\�"�\����Cv��{������Q��{Cq��                                    Bxb�,�  �          @�ff�{���R�?\)���
Cu��{���������$ffCp�R                                    Bxb�;B  �          @�Q��
�H��=q�G���=qCx� �
�H��
=����*G�Ct!H                                    Bxb�I�  �          @ڏ\�������
�a�����C~#׿�����z���
=�;  Cz
=                                    Bxb�X�  �          @�(������Ϯ��(��K�C��\��������O\)��G�C���                                    Bxb�g4  �          @ָR=�\)��z�(����C�G�=�\)���!G���p�C�N                                    Bxb�u�  �          @�(������ۅ�#�
���
C��H������=q���R���RC��3                                    Bxb˄�  �          @���>8Q���\)?�@��C��f>8Q����
��\)�;\)C���                                    Bxb˓&  �          @��H?���(�?��A0(�C�aH?���\)�.{���C�XR                                    Bxbˡ�  �          @�p�    ��=q?�G�A
=C��q    �ڏ\�}p���C��q                                    Bxb˰r  �          @�׿z�H��  ����=qC��Ϳz�H����o\)�	  C��q                                    Bxb˿  �          @��
�ff���������RCv�\�ff�\����G��Wz�Cn��                                    Bxb�;  T          @�(��33��ff�h��� �Cyp��33��p�����=�HCs�)                                    Bxb��d  �          @�=q?�{��=q�\)����C���?�{��녿�����Q�C��=                                    Bxb��
  �          @��H@�\)���?��
AO�C��\@�\)���H���
�.{C�5�                                    Bxb���  �          @陚@����{?��HAz�C�+�@�����ÿ����C���                                    Bxb�V  T          @�\)@c33��ff>��@q�C��=@c33��=q��{�.�\C�.                                    Bxb��  �          @��H@(Q����þ�{�0��C���@(Q�������R���C��R                                    Bxb�%�  �          @�=q@0������?aG�@�
=C���@0����(����
��HC���                                    Bxb�4H  �          @��@`  ��z�?���A?
=C�` @`  ������{�3�
C�f                                    Bxb�B�  �          @�p�?�G���\)�#�
��p�C��=?�G������(���(�C�O\                                    Bxb�Q�  �          @��
?��
��ff��R��Q�C�\)?��
��
=�$z�����C���                                    Bxb�`:  �          @߮?�33�أ׾�  ��
C��{?�33�����\)���C�7
                                    Bxb�n�  �          @أ�?��\�Ӆ�@  ��p�C�b�?��\���H�+���p�C��H                                    Bxb�}�  T          @�(��aG��ə���ff�~�\C�  �aG���\)�e��C�ٚ                                    Bxb̌,  �          @�Q�.{��Q쿺�H�I�C�>��.{�����U���\)C�&f                                    Bxb̚�  �          @�z�?�z��׮�u��\C���?�z��˅�\)����C�,�                                    Bxb̩x  �          @�ff?�����=q>���@Y��C��)?�����z�˅�\(�C��q                                    Bxb̸  �          @��?�z���G�>L��?��C���?�z���G����t��C��                                    Bxb���  �          @���?�z��ٙ�?�@�33C�� ?�z���z���
�I�C�f                                    Bxb��j  �          @��
@Q���p�>�ff@h��C��3@Q���  ��=q�Mp�C��                                    Bxb��  �          @��@!G��љ�?�R@���C��H@!G�������5��C��\                                    Bxb��  �          @��
?�����33>��@qG�C��?�����p��У��S\)C��                                     Bxb�\  �          @�@\)��?G�@�
=C�4{@\)�Ӆ���
�$��C�P�                                    Bxb�  �          @陚@N�R��  ?#�
@���C�7
@N�R��z΅{�,��C�j=                                    Bxb��  �          @�G�@dz�����?k�@�C��@dz���Q쿅���HC��
                                    Bxb�-N  �          @��H@|�����?���A
=qC���@|���Å�L����G�C�s3                                    Bxb�;�  �          @�G�@q��\?���A+
=C���@q���ff����  C���                                    Bxb�J�  �          @���@U��z�?�G�@�\)C��=@U��z�}p����C���                                    Bxb�Y@  �          @�Q�@[���Q�?��A
ffC�^�@[���G��c�
��=qC�O\                                    Bxb�g�  �          @�@>�R���?\)@��RC�1�@>�R��p���p��=��C�p�                                    Bxb�v�  �          @�33@=p���z�?c�
@�C�h�@=p���33�����{C�y�                                    Bxbͅ2  �          @�@\(���G�?���A4(�C���@\(���p������ffC��
                                    Bxb͓�  �          @�z�@b�\�\?��AQ�C�)@b�\���
�Y���ۅC��                                    Bxb͢~  �          @�z�@j=q���H?�@�G�C���@j=q��ff�����3
=C���                                    Bxbͱ$  �          @��H@~�R����?L��@У�C�E@~�R��������HC�Z�                                    BxbͿ�  �          @�p�@aG�����?�A�
C�
@aG��Å�E���Q�C���                                    Bxb��p  �          @�ff@I���ʏ\?u@���C�>�@I����녿�����C�E                                    Bxb��  �          @��@��Å@.�RA���C���@����?.{@�\)C���                                    Bxb��  �          @�?��У�@��A��C�޸?�����=���?L��C�u�                                    Bxb��b  �          @�ff@333����?��HA\��C��3@333�Ӆ�\�C�
C�|)                                    Bxb�	  �          @�(�@5�����?�ffAi�C��@5����þ�=q�p�C��
                                    Bxb��  �          @�@8Q��θR?��A�C��@8Q��θR��ff��C�                                    Bxb�&T  �          @��
@"�\��33?J=q@��
C���@"�\��Q쿫��-��C��=                                    Bxb�4�  �          @�G�@���Ϯ?��A+33C�  @����=q�Q���p�C��                                    Bxb�C�  �          @�\)@%��p�?L��@��HC��R@%���H���
�)��C�R                                    Bxb�RF  �          @�ff?�{����?�
=A��C�˅?�{����  �(�C���                                    Bxb�`�  �          @�ff?������?�ffA,��C��?�����\)�c�
����C�\                                    Bxb�o�  �          @�  ?������?���At��C��q?����ٙ���33�5C��                                     Bxb�~8  �          @�
=?ٙ���{?�p�A�Q�C�ff?ٙ���\)�B�\����C�
                                    BxbΌ�  �          @�\)?��
��33?�\)Ayp�C���?��
��33��{�4z�C�O\                                    BxbΛ�  T          @�33?aG��Ϯ?�(�A�Q�C�� ?aG����þaG�����C���                                    BxbΪ*  �          @���?��H��  @�A��
C��{?��H��33�#�
��C�>�                                    Bxbθ�  �          @ָR?Y����ff?ǮAX  C��q?Y���Ӆ�(���ffC��f                                    Bxb��v  �          @׮>�Q�����?���AG\)C���>�Q����Ϳ@  ���C���                                    Bxb��  �          @ָR?z��ʏ\@Q�A�{C���?z����#�
��33C�xR                                    Bxb���  �          @���=L����p�@�A���C�=q=L����Q�#�
��p�C�:�                                    Bxb��h  �          @�z�����\)?�33A���C�]q����ȣ׾�����C�}q                                    Bxb�  �          @�녿��R����?#�
@�z�C��׿��R��Q쿽p��Q�C�e                                    Bxb��  �          @�G��\)��Q�=�\)?��C��\�\)��p���
��p�C�l�                                    Bxb�Z  �          @��Ϳ�p����ÿE���p�C{���p���ff�-p���=qCz�                                    Bxb�.   �          @�{����{�����,��CzQ������R�E���Cw�{                                    Bxb�<�  �          @����\)��Q쿅����C}^���\)��=q�B�\���C{p�                                    Bxb�KL  �          @�{>�=q���
>��@�C�(�>�=q���H��
=��C�5�                                    Bxb�Y�  T          @��?����p�?aG�@��C���?���Å���H�0��C��f                                    Bxb�h�  �          @�G�@��
����@ffA�{C��=@��
��>��
@(Q�C�l�                                    Bxb�w>  �          @�\)@�G����
@G�A���C��{@�G���  >W
=?��HC��3                                    Bxbυ�  �          @ᙚ@�\)���@33A�33C�u�@�\)���
>B�\?��
C�z�                                    Bxbϔ�  �          @ᙚ@~{��33@0��A���C���@~{��
=?fff@�(�C�`                                     Bxbϣ0  �          @�\)@U�����?&ff@���C�q�@U����Ϳ�33�:�RC���                                    Bxbϱ�  �          @�\)@Z=q��
=?k�@�z�C��)@Z=q��{�������C��                                    Bxb��|  �          @�
=@p  ��ff@
�HA�\)C�>�@p  ���
>aG�?�ffC�H�                                    Bxb��"  �          @�z�@vff���@�A�
=C��R@vff��ff>8Q�?��RC��                                    Bxb���  �          @��H@tz���z�@�A�Q�C�Q�@tz���z�?   @��RC��                                    Bxb��n  T          @��@�Q���(�@33A��C��@�Q���33>�G�@g�C�˅                                    Bxb��  T          @�ff@��\���\@=qA�=qC�b�@��\���H?��@�G�C��                                    Bxb�	�  �          @�p�@�G���p�@0��A�Q�C�K�@�G����\?��\A�C�l�                                    Bxb�`  T          @�33@����\)@p�A��HC��H@����ff>��@~�RC�(�                                    Bxb�'  T          @��@�������@9��AƸRC��@�����  ?���AffC��                                    Bxb�5�  �          @��H@����(�@p�A�G�C��q@����?&ff@��RC�c�                                    Bxb�DR  �          @���@�G���ff@(Q�A��C���@�G�����?J=q@���C�                                    Bxb�R�  �          @�p�@o\)��{@#33A��C���@o\)����?5@��HC��                                    Bxb�a�  �          @�
=@{���p�@0  A��C�@{����\?z�HA
=C�/\                                    Bxb�pD  �          @���@Tz�����?�(�Ag�
C���@Tz���Q쾸Q��A�C�u�                                    Bxb�~�  �          @��@l�����?��A:ffC��R@l�������#�
����C�H�                                    BxbЍ�  �          @�z�@j�H��z�?�  AH��C�� @j�H��녿
=q���RC�)                                    BxbМ6  �          @�z�@o\)��{?uA ��C��f@o\)��p�������C��\                                    BxbЪ�  �          @�33@dz���?��A-�C�{@dz����׿@  ���HC��q                                    Bxbй�  �          @�33@h������?��RA'�C�^�@h����\)�J=q��z�C�/\                                    Bxb��(  �          @ۅ@��H���\?��A:�HC��H@��H��\)�����=qC�^�                                    Bxb���  
�          @��
@n{��(�?�z�A�C���@n{���^�R��=qC��H                                    Bxb��t  �          @أ�@�  ��
=?�z�Ac�C�Ff@�  ��\)�8Q��  C���                                    Bxb��  �          @�\)@�G����\?��Ac33C�AH@�G����\�k����RC��{                                    Bxb��  �          @�ff@\)��
=?���A5p�C���@\)��33�
=���C�Y�                                    Bxb�f  �          @׮@�Q���Q�?��\A-G�C���@�Q����
�(����33C�^�                                    Bxb�   �          @�p�@y�����?�=qA7�
C�W
@y�����
�
=���
C�                                      Bxb�.�  �          @�{@p  ��=q?��AT��C��R@p  ���׾�(��l(�C�R                                    Bxb�=X  �          @�33@1����@��A��C�j=@1���>8Q�?�{C��=                                    Bxb�K�  �          @�\)@����\)?�G�Ar�\C���@�����ý�G��s33C�                                      Bxb�Z�  �          @ҏ\@dz���z�@�\A�{C�]q@dz�����=�Q�?@  C�g�                                    Bxb�iJ  �          @���@�������?��A5C��@�����������G�C�c�                                    Bxb�w�  �          @ٙ�@�33��\)>�(�@hQ�C�` @�33��녿�{�9�C��                                     Bxbц�  �          @��@����\>\)?�\)C�S3@�������
=�`��C�%                                    Bxbѕ<  
�          @��H@��
��Q�?L��@��C��{@��
��
=����{C��{                                    Bxbѣ�  �          @θR@e���p�?�p�AS\)C�` @e���33�����
C��f                                    BxbѲ�  �          @�@S�
���\?���AG
=C���@S�
��
=�(����RC���                                    Bxb��.  �          @��H@9�����R?�{AF�HC���@9�����H�.{��p�C���                                    Bxb���  �          @��
@@  ��G�?\(�@�G�C�4{@@  ��\)��
=�+\)C�U�                                    Bxb��z  �          @��@W����?�z�A*=qC��{@W���\)�E���=qC�ff                                    Bxb��   �          @��
@Y������?^�R@��
C�T{@Y����Q쿋���\C�n                                    Bxb���  T          @θR@u����?�  A3\)C��\@u��z�#�
���RC�`                                     Bxb�
l  T          @�  @a���  ��\����C��q@a���\)����ffC�                                      Bxb�  �          @У�@\(���ff����p�C�c�@\(��o\)����\)C�N                                    Bxb�'�  �          @У�@u����
���H���RC�
@u��u��e�Q�C���                                    Bxb�6^  �          @Ϯ@|(���Q쿗
=�((�C�)@|(�����:�H��{C�y�                                    Bxb�E  �          @�  @r�\������  �V{C�p�@r�\����O\)��33C�*=                                    Bxb�S�  T          @�G�@`  ��G���z��N�RC�h�@`  ���H����C��q                                    Bxb�bP  �          @��
@W�����@FffB
=C��@W��4z�@{A֣�C�                                      Bxb�p�  �          @��@H�ÿ��@��BJ�C�e@H���1G�@X��B�HC�AH                                    Bxb��  �          @�  @R�\���\@�33BEG�C�xR@R�\�+�@VffBG�C�j=                                    BxbҎB  �          @�(�@.�R�1�@e�B*Q�C�5�@.�R�y��@�
A��C�y�                                   BxbҜ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxbҫ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxbҺ4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�׀   �          @���@�p���\)?�33A/\)C�K�@�p���33���H��ffC��H                                    Bxb��&  �          @���@HQ���\)?�=qA��HC�y�@HQ�����B�\��Q�C��{                                    Bxb���  �          @���@����=q?�A���C��@��������
�B�\C�E                                    Bxb�r  �          @�?�����33@ffA�=qC���?�������>��?�ffC�l�                                    Bxb�  "          @��H?������?��HA���C�>�?�����=#�
>��C��=                                    Bxb� �  �          @��?�Q���z�?�=qA�
=C�f?�Q��������z�C�`                                     Bxb�/d  
�          @�  ?��H���R?O\)A  C�Ф?��H��(���Q��IC���                                    Bxb�>
  "          @�p�?�����>.{?���C�P�?�����
=���C���                                    Bxb�L�  
Z          @ƸR?�\)��G�<#�
>�C�g�?�\)��33�  ��Q�C��H                                    Bxb�[V  T          @ə�?�����z�<#�
=�\)C�5�?�����ff�33��\)C��                                    Bxb�i�  T          @�=q?���=q�����{C��\?���=q��H���RC�W
                                    Bxb�x�  "          @�Q�?�����=#�
>���C�e?���  ���C��                                    BxbӇH  
�          @��?�(����þǮ�o\)C�#�?�(����R�"�\���C���                                    Bxbӕ�  �          @�=q>��H��Q�?�A�\)C�h�>��H���׾��H��C�N                                    BxbӤ�  �          @��?z���p�?�p�A���C��?z�����
=q���C�˅                                    Bxbӳ:  "          @�
=��Q���(�?��HA���C�����Q���ff���
�C33C��
                                    Bxb���  �          @�
=�B�\���
?p��A&{C��þB�\���H�����>ffC��{                                    Bxb�І  �          @�=q?������?aG�AG�C�|)?�������
=�Dz�C��3                                    Bxb��,  �          @�p�?�
=���׾.{��C��?�
=��33��p���Q�C��q                                    Bxb���  
�          @�Q�@���zὸQ�z�HC�1�@����ÿ�  ��ffC�AH                                    Bxb��x  �          @���?У����W
=�ffC��{?У���Q��$z����C��                                    Bxb�  �          @�{?�=q��
=�Y���
=C�&f?�=q�s33�\)��(�C��)                                    Bxb��  T          @��@���Q�#�
��
=C�Q�@��l(��{�י�C�)                                    Bxb�(j  �          @�\)@
=�����
���RC�e@
=�QG��AG��  C�ff                                    Bxb�7  �          @��\@�e�j�H�%�
C�  @��33��Q��m=qC�ٚ                                    Bxb�E�  �          @��\?��~�R�S�
�Q�C��?�������H�`��C�Y�                                    Bxb�T\  T          @��
?�
=�u�QG���C�˅?�
=��\����f��C�!H                                    Bxb�c  "          @��?�
=��ff�   ��
=C��?�
=�g
=��
=�=��C���                                    Bxb�q�  "          @���@ff���\��p���  C�n@ff�z�H�r�\���C�}q                                    BxbԀN  T          @Ϯ@���G��L����z�C��@���Q��E���G�C�=q                                    BxbԎ�  �          @У�@����ÿ���!�C���@����
�X������C�C�                                    Bxbԝ�  �          @�@(����ü��
�L��C���@(������
=����C��                                    BxbԬ@  
Z          @�=q@z���?=p�@�  C�@ @z������(��r�RC���                                    BxbԺ�  �          @�  ?����Q�?��
Ac\)C��?�����ͿTz���C��3                                    Bxb�Ɍ  
(          @���@
�H���R?Q�@��
C��@
�H��녿����c
=C�@                                     Bxb��2  
�          @�=q@(����\?�  A�\C�C�@(���Q쿬���E�C�aH                                    Bxb���  �          @�p�?��\��(��=p����C���?��\�Fff�����V�C�"�                                    Bxb��~  "          @�ff?˅��(������\)C��R?˅�����j=q� �
C���                                    Bxb�$  "          @�
=?��R��
=�˅��=qC��{?��R���`���(�C�˅                                    Bxb��  �          @��?����=q������RC�8R?�����u��=qC�0�                                    Bxb�!p  T          @Ϯ?�
=�Å�s33��\C�Ǯ?�
=����QG�����C��                                    Bxb�0  �          @θR@  �����8Q��\)C�1�@  ����#33��  C�"�                                    Bxb�>�  �          @�=q@(���p��#�
���C��@(�����%����C��=                                    Bxb�Mb  �          @˅?�z���\)?��A(�C��q?�z������z��N�\C���                                    Bxb�\  �          @�(�?�{��=q?�33AJ�HC�U�?�{��(���{� ��C�Ff                                    Bxb�j�  T          @��?�\)���?333@��C�(�?�\)�������=qC�Y�                                    Bxb�yT  �          @�=q>�G�����>�@�ffC�*=>�G�����������RC�Ff                                    BxbՇ�  �          @���>�(���=q?\(�A�HC��>�(�����G��n=qC�+�                                    BxbՖ�  "          @��?.{��p�?�G�A z�C�g�?.{�����=q�S�C�q�                                    BxbեF  T          @��?aG���=q?���A\z�C���?aG�����fff��C��q                                    Bxbճ�  
�          @���@
=���?�@��C���@
=��z��{��\)C�,�                                    Bxb�  T          @�p�@$z�����>L��?�p�C�}q@$z���ff��=q���C�P�                                    Bxb��8  �          @��@n{����?�z�A>�HC�j=@n{���Ϳ
=q���\C��R                                    Bxb���  �          @�  @K�����?�p�A�33C���@K���
=�8Q���C��R                                    Bxb��  T          @�Q�?�����ff>#�
?˅C��\?���������\���RC�P�                                    Bxb��*  T          @���>Ǯ��  ?
=@��\C��3>Ǯ��  ��\��{C��                                    Bxb��  �          @Å?޸R��33?�\A��\C��?޸R���H�!G����C�N                                    Bxb�v  �          @�?�\��{?��RA`��C��H?�\��녿k��
=qC�u�                                    Bxb�)  �          @�  ?޸R���R?�G�A���C��R?޸R��ff�z����\C�|)                                    Bxb�7�  I          @ƸR?�����  @ ��A���C�s3?����\������C��                                    Bxb�Fh  
�          @�{?�  ���R>���@w�C�\)?�  ��(���Q���33C���                                    Bxb�U  
�          @�?�  ����>#�
?\C�Ff?�  ���R�\)��ffC��f                                    Bxb�c�  �          @�
=?�����?xQ�A��C��3?�����R��(��\��C���                                    Bxb�rZ  T          @�z�?�����H?�{A�Q�C��{?�����
����=qC�\)                                    Bxbց   �          @�p�?��H����@�A���C��
?��H�����z��1G�C�h�                                    Bxb֏�  �          @Å@G���33>�{@O\)C�@G���  ���H��p�C��{                                    Bxb֞L  @          @���?ٙ���{?��@���C�L�?ٙ���ff��\���
C���                                    Bxb֬�  T          @�
=?޸R��33?���A.�\C��H?޸R��=q��  �A�C��=                                    Bxbֻ�  T          @��?�����z�?�
=A�G�C���?������\�E���=qC�c�                                    Bxb��>  T          @��?z�H��ff=u?��C���?z�H��ff�Q����
C�3                                    Bxb���  �          @�\)?����>�  @�C��=?����Q������G�C��q                                    Bxb��  T          @��>��
��\)�k��z�C��3>��
��(��#�
�ә�C�                                    Bxb��0  T          @����z���33������C��H��z����
�3�
��C�H�                                    Bxb��  �          @�녾����녿�33�b=qC�8R�������c�
�z�C��                                    Bxb�|  T          @���W
=��
=����pz�C��þW
=���\�p  ��C��{                                    Bxb�""  
�          @��>�=q��{����  C�^�>�=q������(�C���                                    Bxb�0�  �          @��R?�  ��G�=�\)?@  C�#�?�  ���\�p���G�C���                                    Bxb�?n  	�          @���?!G���ff�.{�޸RC�#�?!G����
� ���υC�~�                                    Bxb�N  
�          @�z�@���G�?��\AH��C��\@����H���
�"�\C��
                                    Bxb�\�  "          @�{@���=q?�p�AUG�C��@������ff��RC��q                                    Bxb�k`  
�          @θR@7
=��(�@
=A�
=C�H@7
=���׾����,��C�/\                                    Bxb�z  ?          @��H@�H��  @0��A�33C��\@�H��{>�G�@���C���                                    Bxb׈�  �          @��
@{��p�@{A�C�:�@{����k��33C�k�                                    BxbחR  �          @��@ff���@	��A�{C��q@ff��=q��z��+�C�H                                    Bxbץ�  �          @���@Mp���
=@U�B �\C���@Mp���
=?��\A;�
C��=                                    Bxb״�  �          @ʏ\@8����{@\��B�C��=@8����
=?��
A;�C��                                    Bxb��D  �          @Ǯ@Fff��{@EA��
C�u�@Fff���\?uAC��                                    Bxb���  �          @�(�@S33��\)@HQ�A�z�C�0�@S33��(�?xQ�A�
C��f                                    Bxb���  �          @�@B�\��{@)��A�G�C�z�@B�\���
>�ff@�ffC���                                    Bxb��6  �          @�
=@333��33@
=qA�33C�Y�@333��G��.{�ǮC�b�                                    Bxb���  �          @�p�@ ������@z�A�Q�C�7
@ ����z��G��z�HC���                                    Bxb��  �          @�z�@������?��A���C���@�����H�(���  C�`                                     Bxb�(  T          @ȣ�@=p���?�A��RC��@=p���Q��(��|(�C�#�                                    Bxb�)�  �          @��@e��z�@��A�p�C�� @e���>���@FffC�ٚ                                    Bxb�8t  �          @���@}p�����@(�A�\)C���@}p���{>u@�C��                                    Bxb�G  �          @�z�@�����H?��A��C��)@����Q�#�
��Q�C�Z�                                    Bxb�U�  �          @�z�@~{����?�{Ar�RC�H@~{��p������n{C�+�                                    Bxb�df  �          @��@r�\����?�\)At(�C���@r�\������ff���C�#�                                    Bxb�s  �          @�=q@5��p�?z�HA�HC�ff@5��33����H(�C���                                    Bxb؁�  �          @���@4z���  ?5@�Q�C�"�@4z���녿�=q�t(�C��
                                    BxbؐX  �          @��@a�����?ǮAr�\C��3@a����׿�\����C�AH                                    Bxb؞�  �          @�\)@Fff���?��\AD��C�!H@Fff��\)�n{�z�C��\                                    Bxbح�  �          @\@y����
=?��A'\)C��3@y����Q�fff�	C�p�                                    BxbؼJ  T          @��\@#33���H@G�A��HC�>�@#33���ü#�
���C��                                    Bxb���  T          @��@��H�N�R?�33A�=qC��R@��H�g�=�?�p�C�:�                                    Bxb�ٖ  �          @��@�Q��*=q?��
Az{C�k�@�Q��C33>�  @!�C��=                                    Bxb��<  �          @�(�@��R��H?��A&�\C���@��R�(�ý��
�@  C��                                    Bxb���  �          @�\)@�33��Q�?:�H@�\)C�q�@�33�z�.{���C��H                                    Bxb��  �          @�\)@�=q���H?��A�C�K�@�=q�{=���?xQ�C�%                                    Bxb�.  �          @�  @��H�.{?333@���C�*=@��H��ff?\)@�p�C�˅                                    Bxb�"�  �          @�ff@��H�\?���A0  C�e@��H��{>�p�@e�C�Ф                                    Bxb�1z  �          @�  @��\�c�
?p��A��C���@��\���H>��@��C�"�                                    Bxb�@   �          @�
=@��ÿ�z�?��A��RC��@����!�?c�
A	�C�/\                                    Bxb�N�  �          @���@��H��{?�=qAMG�C�P�@��H�aG�?��A"=qC��{                                    Bxb�]l  �          @�G�@�ff��
=?��A��C���@�ff�ff?�=qA(Q�C�t{                                    Bxb�l  �          @�  @�=q��(�?�ffApz�C��)@�=q���R?E�@��
C�(�                                    Bxb�z�  �          @�\)@�=q�333?���A%�C���@�=q���?+�@�{C���                                    Bxbى^  �          @�Q�@�
=�.{?
=q@��
C�4{@�
=�\>��@{�C�,�                                    Bxb٘  �          @�\)@���>Ǯ@s33C���@����>��?�Q�C�R                                    Bxb٦�  �          @��@�33���?(�@�33C���@�33����=���?xQ�C�5�                                    BxbٵP  �          @�ff@��ÿ�(�?.{@ҏ\C�\@��ÿ��=�Q�?^�RC�<)                                    Bxb���  �          @��R@�G����ͽ�Q�W
=C�n@�G���
=�(����z�C�<)                                    Bxb�Ҝ  �          @�ff@�  �p��L�;�G�C��@�  ��(���  ��RC�%                                    Bxb��B  �          @��
@����*=q?�{AW�C�f@����?\)=�\)?:�HC���                                    Bxb���  �          @��@���J�H?p��A��C�L�@���QG�������C��                                    Bxb���  �          @��
@��
�+�?k�AQ�C�4{@��
�3�
�����QG�C��H                                    Bxb�4  �          @���@�p��J=q?���A�  C�4{@�p��l(�>\@tz�C��                                    Bxb��  �          @��@����8Q�@
=A�{C��=@����`��?&ff@�Q�C�q                                    Bxb�*�  T          @�(�@�Q��E?��HA@��C���@�Q��S33��  �   C���                                    Bxb�9&  �          @��
@�  �l(�?�@�33C�L�@�  �c33����4  C�ٚ                                    Bxb�G�  �          @��@�Q��?\)>��@'�C��@�Q��333�����/�C��                                     Bxb�Vr  �          @�z�@h�ÿ�ff��
=�8p�C��@h��>k����\�M�
@c33                                    Bxb�e  �          @�G�@:�H�G�����n��C���@:�H?Ǯ��\)�a�A�ff                                    Bxb�s�  �          @��H@2�\�G����\�u�HC�=q@2�\?У���(��g33A�Q�                                    Bxbڂd  �          @��H@I��>�=q�����i�@�z�@I��@\)��Q��A\)B�\                                    Bxbڑ
  �          @�p�@��8Q������HC��q@�?�
=���\�p�B�                                    Bxbڟ�  "          @��@
=�j=q�q��#��C�n@
=�У���  �pz�C���                                    BxbڮV  
�          @��\@(��(�������S�RC��
@(��\���  C�
=                                    Bxbڼ�  
�          @���@e�2�\@3�
A��C�\@e�o\)?�ffA`��C���                                    Bxb�ˢ  �          @�z�@�z��1G�@)��A�=qC���@�z��i��?�A8��C�&f                                    Bxb��H  �          @��@���>{@(�A�Q�C��@���n�R?c�
A(�C��
                                    Bxb���  "          @�33@��C33@$z�A�\)C��@��w�?xQ�A�C���                                    Bxb���  "          @��R@�Q��Tz�+����
C���@�Q��,(��z���C�Ff                                    Bxb�:  �          @�Q�@��\�HQ�>�@�=qC�.@��\�AG��s33�=qC��H                                    Bxb��  "          @�G�@�33�9��@�A�ffC��H@�33�b�\?�R@�z�C�5�                                    Bxb�#�  
Z          @��H@�\)�)��?�(�A;�C���@�\)�:�H���
�@  C�j=                                    Bxb�2,  �          @�ff@�\)�#33@A�Q�C���@�\)�Mp�?@  @߮C�7
                                    Bxb�@�  �          @�G�@�(�� ��?�R@�C�.@�(��������I��C��                                     Bxb�Ox  
�          @\@�ff���>�G�@�{C�Ф@�ff�����
�L��C�l�                                    Bxb�^  �          @��@��þ�Q�>�\)@#�
C�` @��þ�G�=�Q�?\(�C���                                    Bxb�l�  �          @��@��H?0�׾��
�8��@Ǯ@��H?B�\<��
>#�
@�(�                                    Bxb�{j  
�          @�
=@ƸR?�녿�ff�G�AJ=q@ƸR?��H���
�6ffAv�R                                    Bxbۊ  �          @Ϯ@�{?�z῝p��/33AMp�@�{?�������RA�Q�                                    Bxbۘ�  �          @�p�@�Q�?��ͿL����ffA�H@�Q�?���k��AA�                                    Bxbۧ\  "          @ƸR@���?�ff����Q�AK
=@���?�p�������A���                                    Bxb۶  
�          @�(�@��?E��k���@��
@��?J=q>\)?�  @�=q                                    Bxb�Ĩ  �          @�p�@�33@�H?�G�A��A\@�33?���@)��A�33Ag�                                    Bxb��N  "          @�p�@��@8Q콸Q�Q�A��
@��@'
=?���A6ffA�{                                    Bxb���  �          @�p�@�33@z��=q��z�Aî@�33@8�ÿ�����A�ff                                    Bxb��  
�          @��
@{�?�ff�fff�=qA���@{�@N{��H��  B33                                    Bxb��@  
Z          @�33@�(�@{�%���
A�R@�(�@W����H�?33B
=                                    Bxb��  T          @�z�@�?p���XQ����A5�@�@ff�&ff��
=A��                                    Bxb��  
�          @�z�@��
�G���33�1ffC��q@��
?��������,��Aup�                                    Bxb�+2  "          @�p�@�33=��~{�)Q�?�ff@�33?���`  ���A���                                    Bxb�9�  
�          @�=q@��
?5�S�
�{A��@��
@��(����p�A�
=                                    Bxb�H~  �          @�G�@�(�?Tz��:=q���A�@�(�@�\��R��33A��
                                    Bxb�W$  �          @�{@�������4z���G�C�b�@��?��$z���AX��                                    Bxb�e�  T          @�33@�z὏\)�E�� ��C���@�z�?����1���{AqG�                                    Bxb�tp  T          @�  @�Q���������p�C�/\@�Q�E��p��θRC��                                    Bxb܃  �          @�\)@��
�r�\���
�P  C���@��
�P  ���R��  C�޸                                    Bxbܑ�  T          @���@��ͿL���5��C�` @���?\)�8����
=@У�                                    Bxbܠb  �          @�(�@��H�C33��33���C�.@��H��p��6ff��C�O\                                    Bxbܯ  �          @���@{����R�L��� ��C���@{��j�H�z�����C��                                     Bxbܽ�  �          @��R@��\�&ff��ff�{�C���@��\�У��#33�ԣ�C��3                                    Bxb��T  l          @�Q�@�Q쿆ff��p����C��q@�Q�#�
�\)���C��f                                    Bxb���  y          @��
@��R�"�\����T(�C�t{@��R����
���HC���                                   Bxb��  �          @��@��R�0  ��  �v�\C��)@��R���
�%���33C��=                                    Bxb��F  [          @�ff@�  �1G��p���C�,�@�  ��p��N�R�	{C��=                                   Bxb��            @�G�@��\�*�H>�G�@��C��
@��\�%��O\)����C�R                                    Bxb��  Q          @�Q�@��H��\?+�@�ffC���@��H�Q쾣�
�EC���                                    Bxb�$8  �          @�
=@�33�7�?�{A,  C�H�@�33�Dzᾞ�R�=p�C�z�                                   Bxb�2�  �          @�\)@�
=�,��?�  A
=C�P�@�
=�7
=�����H��C��                                     Bxb�A�  	9          @�Q�@����33?�\@���C��)@���������vffC��q                                    Bxb�P*  
�          @��@���%@��A�G�C�R@���U�?W
=Ap�C��=                                    Bxb�^�  T          @��R@�{�AG�?�  AC33C�J=@�{�P�׾�  ��C�L�                                    Bxb�mv  �          @�p�@��
�C�
?�=qAP  C��\@��
�U��L�Ϳ�(�C��3                                    Bxb�|  
�          @�Q�@�ff�#33����"�HC��@�ff��ff�����C�p�                                    Bxb݊�  T          @�33@�  ��(��5���p�C�w
@�  >L���Dz���\@��                                    Bxbݙh  
Z          @���@����  �&ff��\C��q@���Q녿�33�h  C���                                    Bxbݨ  
�          @�p�@g��O\)@N{B(�C��@g���33?�33A^{C��f                                    Bxbݶ�  T          @�@��H�XQ�@&ffA�ffC�8R@��H��{?G�@�G�C�(�                                    Bxb��Z  �          @�@�  �j=q?�33A���C��q@�  ��(�<�>�  C��                                    Bxb��   
�          @�33@���aG�?�{A�(�C��@���w��.{��33C�9�                                    Bxb��  �          @�z�@��\�e?�Ab=qC�#�@��\�u��p��l��C�/\                                    Bxb��L  "          @��@���vff>�
=@�\)C��)@���g������`��C�~�                                    Bxb���  �          @�33@�Q쿵�C�
��\C�G�@�Q�>��W���?�                                    Bxb��  �          @�(�@��A�        C�33@��,�Ϳ����\��C��                                    Bxb�>  �          @���@����׾#�
�\C��@��`  ��(���{C��{                                    Bxb�+�  �          @\@����\)����z�C���@���z��=p���p�C��f                                    Bxb�:�  T          @��H@�����c�
���C�C�@��?�G��N{� ffA��H                                    Bxb�I0  �          @\@��H�!녿�\)�+�C�S3@��H�޸R�	�����C���                                    Bxb�W�  �          @�(�@��H�E����
�U�C�@��H�&ff�ٙ����\C�AH                                    Bxb�f|  T          @�p�@�G��J�H?8Q�@��
C��\@�G��H�ÿO\)���C���                                    Bxb�u"  
�          @��@����@��?��@���C�.@����<�ͿW
=�=qC�q�                                    Bxbރ�  
�          @�Q�@����.�R�\)���C���@����
�H�޸R��\)C�O\                                    Bxbޒn  
�          @�ff@�33�<�;����@��C�S3@�33�\)��\)��=qC�h�                                    Bxbޡ  "          @�(�@���G�������C��@���0�׿��H�p��C��{                                    Bxbޯ�  T          @���@���Y��<��
>.{C�l�@���A녿���~ffC��)                                    Bxb޾`  "          @�{@����&ff�k��ffC�J=@����p���33�e�C�                                      Bxb��  J          @�Q�@�G���
=������C��@�G����
�����4��C��H                                    Bxb�۬  
�          @��@�ff�ٙ�=�Q�?Y��C�Q�@�ff��ff�5��33C��                                    Bxb��R  
�          @�Q�@�G���?���A,  C�f@�G����=�\)?.{C���                                    Bxb���  "          @�\)@��R�/\)>�@��C���@��R�(�ÿY���	�C���                                    Bxb��  
�          @�@�G��G
=?Tz�A	�C�Ǯ@�G��I���0������C���                                    Bxb�D            @��@��xQ�>.{?޸RC���@��a녿У�����C��                                    Bxb�$�  	j          @�{@�Q��~�R?   @��HC��q@�Q��p�׿����a��C�g�                                    Bxb�3�  "          @��\@u�=p�@p�A���C�*=@u�p��?Q�A
�RC��=                                    Bxb�B6  �          @��\@����H@�Aә�C�#�@���QG�?��A0z�C�1�                                    Bxb�P�  T          @���@������?ٙ�A�  C��q@����<��>\@vffC�u�                                    Bxb�_�  
�          @�
=@�G��<(�?fffAG�C�=q@�G��AG��\)��33C��H                                    Bxb�n(  �          @��
@��H��k��  C���@��H��
=���
���HC�%                                    Bxb�|�  �          @�\)@�  �%>�33@aG�C�H�@�  �p��c�
���C���                                    Bxbߋt  T          @�G�@�=q�?\)?8Q�@�\)C�)@�=q�>�R�=p����C�                                      Bxbߚ  �          @��@����\)?���Ap��C�7
@����8Q�>\)?�z�C�s3                                    Bxbߨ�  
Z          @�  @��\��  ?5@ۅC�'�@��\���H>.{?�z�C�!H                                    Bxb߷f  T          @�G�@�\)�\)>\)?��C�T{@�\)�B�\=u?�C��                                    Bxb��  
�          @\@�G�?�
=?h��A
{A^=q@�G�?Y��?��HA`��A{                                    Bxb�Բ  
�          @�(�@�{=#�
?�33AT(�>��@�{�(�?�G�A>�HC�{                                    Bxb��X  �          @��
@���=���?@  @�=q?u@��þ��?5@ָRC���                                    Bxb���  
�          @��
@�z�?z�H��33�.�RA33@�z�?�33�����
AV=q                                    Bxb� �  �          @�p�@�=q?�
=��z����RA�G�@�=q@{�z����AǙ�                                    Bxb�J  T          @��@�33�У������HC���@�33���H�&ff��z�C�!H                                    Bxb��  
�          @���@���?���AG��  A��
@���@Dz��ff����B��                                    Bxb�,�  
�          @��@�Q�=�G��=q����?�33@�Q�?��H����AN�R                                    Bxb�;<  "          @���@�Q�?n{��R���A(��@�Q�?�p���\���A�{                                    Bxb�I�  "          @���@��?�������Al(�@��@  �\�{�
A�33                                    Bxb�X�  J          @��@xQ��e���{��  C���@xQ���H�E��  C��q                                    Bxb�g.  �          @���@����Mp����i�C��@����(��0  �陚C��H                                    Bxb�u�  �          @�(�@��H�:=q�����^ffC�Ф@��H���H�"�\��C�t{                                    Bxb��z  
�          @�@�  �1녿�(��G33C�Ф@�  ��z�����z�C�3                                    Bxb��   �          @�p�@`����\)��ff�UG�C��R@`���I���G
=�C��                                    Bxb��  |          @���@I����=q��
=���C��@I���C33�_\)�=qC��
                                    Bxb�l  
d          @��R@l�����׿��H�o�
C�T{@l���8Q��J=q��RC��                                    Bxb�  T          @���@�{�@  �\�tz�C��\@�{���H�/\)���C��\                                    Bxb�͸  T          @�ff@�=q�7
==�?�  C��H@�=q�%��(��H��C��
                                    Bxb��^  �          @�Q�@��<��>L��?��HC��\@��,�Ϳ�Q��?�C��H                                    Bxb��  "          @�  @�33�AG�?   @�Q�C�@�33�:=q�u��C��H                                    Bxb���  
�          @���@�
=�-p�?J=q@���C���@�
=�1G��\)��z�C�s3                                    Bxb�P  
�          @���@�G���p�?s33A\)C�C�@�G���ͽ��
�L��C�8R                                    Bxb��  �          @���@����?�Q�A��
C�{@���=p�?#�
@�
=C�,�                                    Bxb�%�  "          @���@�(��'
=@z�A�{C�L�@�(��QG�?�R@�  C�aH                                    Bxb�4B  �          @��\@�33�A�@�
A�(�C��@�33�h��>�
=@��C�                                    Bxb�B�  
�          @�z�@���u�?fffAG�C�\)@���s�
�z�H�p�C�n                                    Bxb�Q�  J          @��@mp���\)?J=q@�\)C�Ф@mp����H�����Y��C�G�                                    Bxb�`4  
�          @���@�z���z�>��@���C���@�z��xQ��G��n{C�n                                    Bxb�n�  �          @�(�@�����\)?�@�=qC���@�����Q쿺�H�f�\C���                                    Bxb�}�  "          @��@Vff�����Ǯ�vffC�t{@Vff��=q�%���z�C��)                                    Bxb�&  �          @��@P������>��@���C��)@P����\)��Q����RC��                                    Bxb��  "          @��@7����H��\)�[\)C�Z�@7��i���]p���HC�q                                    Bxb�r  �          @�33?�����R�7���C�}q?���C�
��33�`33C�@                                     Bxb�  �          @�G�?��
����^�R���C���?��
�|(��C33�
=C��R                                    Bxb�ƾ  
�          @�  @
=q���������0z�C��@
=q�~�R�S33�Q�C�<)                                    Bxb��d  �          @�  @
=��z�+��ָRC�XR@
=����>�R��(�C��\                                    Bxb��
  "          @�
=?������=�\)?333C���?�����z��\)�ә�C�+�                                    Bxb��  T          @�
=@=q���\�
=��  C���@=q��
=�8Q���\)C��                                     Bxb�V  �          @��@B�\���@~�RB9C��@B�\�vff@{A�(�C�"�                                    Bxb��  �          @�G�@��ÿǮ@�\A��
C�` @����(�?�ffAQG�C��                                    Bxb��  
Z          @�  @�
=�P  ?��A��C�1�@�
=�p  >��?�  C�/\                                    Bxb�-H  
�          @��@Y�����?�A��RC��{@Y����{��z��=p�C���                                    Bxb�;�  
�          @���@g
=�u�@�A���C���@g
=���H<��
>8Q�C��H                                    Bxb�J�  
�          @�@^{��?�Q�AD��C��q@^{��  �c�
�  C��                                     Bxb�Y:  "          @��
@Vff��33�E��   C��f@Vff�`  �-p����
C��H                                    Bxb�g�  "          @�
=@c33��ff��(���33C�O\@c33�o\)�����p�C�Ǯ                                    Bxb�v�  
�          @��
@�������?c�
AC�aH@�����  �����/33C���                                    Bxb�,  
�          @�z�@����}p��W
=�33C��H@����Z=q�G���z�C��                                     Bxb��  
�          @��@`�����ÿ^�R� ��C�q�@`�������G
=��
=C�s3                                    Bxb�x  �          @�ff@����33�0����ffC���@���p  �0  ��Q�C���                                    Bxb�  "          @�z�?�  ��������6
=C���?�  ��  ����C��q                                    Bxb��  T          @Ϯ?#�
��=q����=z�C�/\?#�
��\)�ƸR�RC��R                                    Bxb��j  �          @��H?�����������(  C�B�?�������
��C���                                    Bxb��  T          @��H?��
������Q��.�C��H?��
��
=�Åz�C��3                                    Bxb��  �          @ָR?����
=�����CG�C�~�?�������#�C�l�                                    Bxb��\  �          @�(�?�
=�O\)����]�HC���?�
=�����ƸR��C�:�                                    Bxb�	  	�          @��H@ �׾�  ��{aHC�o\@ ��@"�\��(��k�RBN�R                                    Bxb��  �          @ʏ\?�  ���������\C�O\?�  ?�������HBPQ�                                    Bxb�&N  �          @�\)��G�����(��p�C�y���G��(���ff�qC��                                    Bxb�4�  
�          @ҏ\=�\)�s�
���\�Y�\C��=�\)�fff�Ϯ¤.C�1�                                    Bxb�C�  T          @�p��E��(���z��Cz�H�E�?L����\)��C�q                                    Bxb�R@  �          @ƸR?�G���\)����=qC�)?�G������3�
�؏\C�H                                    Bxb�`�  "          @���?�{���׿����5��C���?�{�����p  �\)C���                                    Bxb�o�  �          @�p�?�ff���H��R��=qC��{?�ff��(��QG�� z�C�#�                                    Bxb�~2  
�          @���?�������J=q����C�� ?�����=q�Z�H��C�*=                                    Bxb��  T          @�{@5����G�����C��H@5���{�N{��G�C�<)                                    Bxb�~  �          @���@L(����\)��G�C��@L(���=q�:=q��\)C�33                                    Bxb�$  �          @���@�Q����
��
=�y��C�y�@�Q��y���!G����C��                                    Bxb��  �          @��@y����\)��
=�|��C���@y����  �%���{C�*=                                    Bxb��p  T          @�G�@<(���
=>��@{�C��\@<(���=q��
����C��3                                    Bxb��  
(          @��@1G���  ?333@�ffC��\@1G���
=�������C���                                    Bxb��  �          @��@4z����>�G�@��RC�!H@4z����H�33���\C�
                                    Bxb��b  "          @�\)@,�����\��G��F=qC���@,���z=q�^�R��
C�N                                    Bxb�  
(          @�  @Tz�����?s33Ap�C��@Tz����׿�Q��]�C�g�                                    Bxb��  �          @��H@s33���H?k�Az�C���@s33��������IC��                                    Bxb�T  
�          @��
@S�
���\?��@�z�C���@S�
���׿�=q���C�\)                                    Bxb�-�  �          @�\)@'
=��\)?E�@�C�Ff@'
=�����  ���C���                                    Bxb�<�  T          @���@@�������#�
��ffC�^�@@�����:=q���C��)                                    Bxb�KF  
�          @��R@Dz���
=�ٙ���(�C�� @Dz��X���l(���RC��                                    Bxb�Y�  T          @��@W���\)�u��C���@W�����\)��  C���                                    Bxb�h�  �          @���@   �����H����z�C��=@   �(����\�XQ�C��{                                    Bxb�w8  
�          @���@0����
=�(���33C��{@0�����H�=p���
=C�=q                                    Bxb��  	�          @��@Vff��33��ff��z�C�P�@Vff���H�)�����C��=                                    Bxb䔄  
�          @�  @|(���\)�L�Ϳ�C���@|(��xQ�������C��q                                    Bxb�*  
�          @�Q�@G������=q���
C�` @G��HQ��l����
C�l�                                    Bxb��  
�          @�  @L�����R��33�Z=qC�h�@L����\)�'
=�ѮC���                                    Bxb��v  
�          @�G�@@  ��G���R���
C��R@@  �7������7��C��                                    Bxb��  T          @�ff@3�
���H�����33C��@3�
�I����p��9
=C��                                     Bxb���  �          @�33@���p����R�L(�C�  @��r�\�XQ����C�U�                                    Bxb��h  "          @�(�@8Q���{��
��
=C�� @8Q��7
=����5Q�C��)                                    Bxb��  �          @��@1G��}p��[��Q�C�t{@1G���
=���
�]G�C���                                    Bxb�	�  
�          @�G�@
=���>�R��=qC��q@
=�"�\��{�[Q�C��{                                    Bxb�Z  
�          @���@g��c33?�AP(�C��@g��l(��(���C�8R                                    