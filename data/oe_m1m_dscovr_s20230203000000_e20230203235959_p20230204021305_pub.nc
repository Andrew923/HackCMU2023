CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230203000000_e20230203235959_p20230204021305_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-04T02:13:05.406Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-03T00:00:00.000Z   time_coverage_end         2023-02-03T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        m   records_fill         3   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxaI*@  �          @��@<(�����<#�
=uC�� @<(���  ����C���                                    BxaI8�  �          @Å@C�
����<�>��
C�@C�
��  �33���C�ff                                    BxaIG�  T          @�=q@QG���33>8Q�?��HC�T{@QG��������C���                                    BxaIV2  T          @���@8Q���(�����]�C�E@8Q��p���Z=q��C��R                                    BxaId�  
�          @���@0  ����=#�
>���C���@0  ������
���C���                                    BxaIs~  �          @�{@!�����?xQ�A�C��{@!���������c33C��                                    BxaI�$  
�          @�z�@�
��33?Y��A(�C��@�
��p������|��C��                                    BxaI��  �          @�ff@�����?�G�A��C�l�@���G���p��e�C���                                    BxaI�p  
(          @�(�@�\���H?aG�A	C���@�\�������w�C��                                    BxaI�  
�          @��\@����?aG�A�
C��{@���ff�����{
=C��                                     BxaI��  "          @�ff?��R����?p��AG�C��?��R���
�����x��C�&f                                    BxaI�b  "          @�33@�����R�����s�
C��)@����\)�1G���ffC�u�                                    BxaI�  �          @Å@4z����
��G����
C��{@4z���(��0����  C���                                    BxaI�  
Z          @��
@333��p����H���HC��@333�����5���(�C���                                    BxaI�T  T          @�p�@2�\��  ���
�5C�w
@2�\�����\)���\C���                                    BxaJ�  
�          @�(�@0  ��
=<�>���C�XR@0  ���Q����HC���                                    BxaJ�  "          @��H@0����p��L�;�(�C�|)@0����33����(�C���                                    BxaJ#F  �          @��@0  �����\)�0��C�xR@0  ��=q�(���(�C���                                    BxaJ1�  
�          @�G�@$z����R=�\)?0��C���@$z������=qC��H                                    BxaJ@�  Z          @��
@'
=���R�����{�C�J=@'
=��Q��)����  C��                                    BxaJO8  T          @��R@0������\)�ͮC�(�@0���5������9�C�*=                                    BxaJ]�  �          @�G�@9����Q��(Q��؏\C��@9���*�H����<Q�C���                                    BxaJl�  T          @�
=@!G���33�Dz��(�C�Ф@!G�������S��C��H                                    BxaJ{*  �          @��\@\)���H�AG����C�XR@\)�ff��Q��X��C��                                    BxaJ��  �          @��\@���(��=p�� �C�o\@��=q��
=�U�
C��                                    BxaJ�v  
�          @�G�@�����
�<��� p�C�c�@���=q��ff�U��C��)                                    BxaJ�  �          @�G�@$z����
�)������C�  @$z��"�\��{�D��C���                                   BxaJ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxaJ�h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxaJ�  �          @��
@�������{C�t{@�)���{��Ez�C�3                                   BxaJ�  �          @��R?�p���Q��0���(�C��H?�p������\)�f  C���                                    BxaJ�Z  "          @��
?�G���z��:�H��C���?�G��(����_=qC��=                                    BxaJ�   T          @��?�p����R�L(��33C�.?�p�������R�g��C��R                                    BxaK�  �          @��?�(����
�=p��C�P�?�(��(����=q�b33C��R                                    BxaKL  
�          @��?�33����"�\�مC�=q?�33�I����33�NffC��)                                    BxaK*�  
�          @��?���
=�#�
��p�C�e?��HQ�����O{C�=q                                    BxaK9�  
�          @��R?�������*=q��
=C�C�?����?\)�����W=qC���                                    BxaKH>  �          @�
=?����33�.�R���C���?���<(���
=�Z��C���                                    BxaKV�  "          @�
=?�Q���\)�{���C�{?�Q��K���G��N�\C�K�                                    BxaKe�  �          @�ff?���ff���
���\C��)?��k��p���.�C��f                                    BxaKt0  
�          @�{?�33��Q��G���p�C�y�?�33�o\)�qG��/��C��\                                    BxaK��  "          @�33?�z���33��\)�AG�C�k�?�z�����N�R�z�C���                                    BxaK�|  �          @��
?�\)���H���\�/�C��f?�\)��z��H����C�'�                                    BxaK�"  T          @��H?�=q���R�aG���C�޸?�=q��33�p���(�C��                                     BxaK��  "          @�Q�?�33��\)�c�
���C�ٚ?�33��33�>{�	�
C�o\                                    BxaK�n  T          @�Q�?�33��=q���
�b=qC�l�?�33�q��P  ��HC��                                     BxaK�  �          @��?�����#�
��RC�"�?���33�,(���(�C��=                                    BxaKں  T          @�Q�?�\)��  �
=����C�\?�\)�����%���HC�k�                                    BxaK�`  
�          @��?Ǯ�������33C�XR?Ǯ�}p��=q��Q�C��H                                    BxaK�  
`          @�=q?�ff��(��W
=��C�y�?�ff�����
����C�k�                                    BxaL�  
Z          @��
?   ��G���33��z�C�q?   �|(������C���                                    BxaLR  T          @�z�Y���?У�B��Cz{�Y���4z�>�Ap�C}=q                                    BxaL#�  �          @��?����ý�G���  C��f?���G���
��G�C��                                    BxaL2�  
(          @��?�
=����>\@���C��{?�
=���׿�G���(�C�+�                                    BxaLAD  �          @��@.{�N�R����
=C�\@.{�9��������C��R                                    BxaLO�  �          @�(�@=q�b�\?B�\A Q�C��@=q�aG��Y���3�C�1�                                    BxaL^�  
�          @��H@��y��>��H@ƸRC�j=@��n�R���R�{\)C���                                    BxaLm6  �          @��?�33����=���?��HC�S3?�33�q녿ٙ����C�Y�                                    BxaL{�  �          @�\)@E�J=q��(���\)C�/\@E�{�,(��{C�.                                    BxaL��  
Z          @��\@*=q�^{�k��=G�C��H@*=q�.�R�z���ffC�)                                    BxaL�(  
�          @�?��R��G�?xQ�AA�C��?��R��G��xQ��A�C��                                    BxaL��  �          @��@�H�Z=q������HC��@�H�
=�`  �>{C�y�                                    BxaL�t  T          @���@(Q��E�!G�����C�7
@(Q�ٙ��fff�C�HC��f                                    BxaL�  
�          @�ff@'
=�*�H�/\)��C�'�@'
=��p��g��M�
C�T{                                    BxaL��  
�          @�@���l�Ϳ�33���C��@���!G��S33�3C��                                    BxaL�f  
�          @��@ ���QG���\�ҸRC��@ �����P  �3�C�4{                                    BxaL�  T          @�Q�?�ff��z�:�H�{C��
?�ff�i���"�\�ffC�|)                                    BxaL��  �          @�{?��
��=q�#�
��C��?��
��G��p�����C��                                     BxaMX  
�          @�  ?0����ff�aG��\)C��3?0����(���
�܏\C�t{                                    BxaM�  
�          @�
=?333������Ϳ�
=C�\?333��z��(����
C���                                    BxaM+�  
�          @�  ?n{���;�z��VffC�ff?n{�����
=��G�C�                                      BxaM:J  �          @�\)?fff��(��k��,��C�>�?fff��=q�33��=qC��=                                    BxaMH�  
�          @�G�?�(����@  ���C��?�(��z=q�*�H� (�C��f                                    BxaMW�  �          @�\)?�p���
=�   ����C��{?�p�����{��  C�f                                    BxaMf<  
�          @�p�?�\)���R��Q���ffC�q?�\)��33����z�C�:�                                    BxaMt�  "          @�  ?�ff��녾�p���
=C��{?�ff��{�Q���C��H                                    BxaM��  
�          @��
?�Q���ff��Q���=qC�� ?�Q����\�(����C��                                    BxaM�.  T          @���?��
���\�@  �	C��?��
����.�R��\C�q                                    BxaM��  
Z          @�=q?�Q����þ�������C�^�?�Q����������C���                                    BxaM�z  
�          @���?�=q��p���=q�tQ�C��R?�=q�j�H�K���
C�                                      BxaM�   T          @��H?����Q쿮{�x��C��H?���o\)�P���33C���                                    BxaM��  T          @���?!G���z�+�����C��?!G���z��+��z�C�J=                                    BxaM�l  �          @�G�?�(���
=��33�U�C�=q?�(��r�\�B�\��C���                                    BxaM�  �          @���?fff����ff��{C�n?fff�e��XQ��)�HC��                                    BxaM��  "          @�z�?k���ff����(�C��\?k��^�R�j=q�5�C�o\                                    BxaN^  
�          @���?����ff�
=��ffC�޸?���A������M(�C��)                                    BxaN  
�          @�Q�?z�H��(������
C���?z�H�4z��n{�M  C��3                                    BxaN$�  "          @�=q>�\)��(�@���B�ffC�e>�\)�J�H@{�BL  C�}q                                    BxaN3P  �          @�Q�?
=q���R@��\B�C���?
=q�u@b�\B)��C���                                    BxaNA�  T          @�=q>�ff���@���B~�C��\>�ff�y��@L(�B=qC�N                                    BxaNP�  �          @���>\)�   @�(�B�
=C��q>\)�p��@VffB&C�\                                    BxaN_B  T          @��\>�ff�)��@���Bg(�C���>�ff��ff@0��B��C�f                                    BxaNm�  T          @�p�>��g
=@k�B5ffC���>���=q?�=qA�\)C��{                                    BxaN|�  "          @�33>����1�@�(�Bp�RC�3>������\@QG�B�
C��                                     BxaN�4  �          @�>�Q��\(�@��BQ�C��
>�Q���  @(��A��
C�
=                                    BxaN��  T          @�{>��u�@�p�B<�HC���>���
=@�A��HC��                                    BxaN��  "          @��
>����c�
@��\BI�C��>�����G�@(�A�(�C��                                    BxaN�&  �          @�
=>W
=�\(�@�Q�BL�C���>W
=����@�A��HC�5�                                    BxaN��  �          @��R>�\)�AG�@���B`�C��)>�\)��z�@6ffA�z�C��{                                    BxaN�r  "          @��\>W
=�@��@�z�B^=qC�H>W
=���@/\)A���C�Q�                                    BxaN�  "          @�(�>L���Z=q@���BJ\)C��{>L�����\@
=A��C�4{                                    BxaN�  
�          @�{=u�_\)@��BP��C�}q=u��G�@(Q�A�ffC�W
                                    BxaO d  	�          @�\)<��b�\@��RBNp�C�B�<���=q@%A�=qC�.                                    BxaO
  T          @��\?=p���\)@5A�G�C�w
?=p���\)?�R@˅C��)                                    BxaO�  
�          @�Q�?333���
@�RA�C�
?333���R>k�@=qC���                                    BxaO,V  �          @��H?^�R����?��A���C��=?^�R��
=��
=���C���                                    BxaO:�  "          @�z�?�  ��ff@&ffA��C�Z�?�  ��(�?�@��C��=                                    BxaOI�  
�          @�?�������@C33B	�HC��?�����ff?���A<z�C��                                    BxaOXH  �          @�ff?������R?�33A3
=C��\?���������K�C��R                                    BxaOf�  
�          @�ff?�p���p�?��\AEC��?�p���{��
=�8  C��                                    BxaOu�  �          @�  ?�����{?��AM�C��\?�����\)�����-G�C�                                    BxaO�:  T          @��?��
��33?���A`Q�C��q?��
��ff�}p��  C�|)                                    BxaO��  
�          @�
=?�
=���?�\)A|(�C�8R?�
=����O\)���\C��                                    BxaO��  �          @�ff?�z����?��HA��
C��q?�z���ff��
=����C�#�                                    BxaO�,  "          @��?�(�����@�A��RC�
=?�(���zᾣ�
�E�C�|)                                    BxaO��  T          @��?����?ǮAw33C���?����33�O\)���C���                                    BxaO�x  T          @�?޸R��G�?��HA<z�C��3?޸R�������5��C��\                                    BxaO�  
�          @�{?�
=���H>�@��C���?�
=������{���C�                                      BxaO��  �          @�(�?�=q���\�u�z�C��?�=q��=q����C��                                    BxaO�j  �          @�=q?��R���R>�?��C�H?��R��G��
=��z�C��                                    BxaP  Z          @���@�R��������z�C�j=@�R��(��(����G�C��                                     BxaP�  T          @��?��H��{>�\)@2�\C��=?��H���H���H��Q�C��                                    BxaP%\  
�          @��?�����>��R@FffC�xR?����z�������RC�\                                    BxaP4  
.          @��\?�Q���\)<�>���C���?�Q����������C��{                                    BxaPB�  
�          @��?�����G��������C���?�����=q�333��{C��                                    BxaPQN  	�          @�ff@����33���H�d  C��H@�������`����RC�7
                                    BxaP_�  
�          @�p�@
=q��=q�p�����C��@
=q����AG���C��)                                    BxaPn�  �          @��?������fff���C�o\?�������C33��Q�C���                                    BxaP}@  "          @�G�?����{����C��=?������.{����C�Ф                                    BxaP��  �          @��?�\)���Ϳ���p�C���?�\)��\)�)����z�C�Ǯ                                    BxaP��  "          @��\?�p���\)�\(��=qC�.?�p���{�9�����\C��                                    BxaP�2  
Z          @��\?������R��z�C���?������+���C��R                                    BxaP��  
(          @��
?�=q��Q�!G���\)C��)?�=q����,����  C��3                                    BxaP�~  
Z          @��?�����ÿ}p�� z�C�XR?����{�A��G�C���                                    BxaP�$  	�          @�ff?��H���׿^�R���C�4{?��H��\)�:�H��z�C��{                                    BxaP��  (          @�z�?�z���p�?   @��C�J=?�z���p���(����\C���                                    BxaP�p  �          @�{?�z���
=?�  A!��C��q?�z�������
�P(�C�                                    BxaQ  �          @��?������?k�AG�C��q?������\��=q�Yp�C��R                                    BxaQ�  T          @�  ?�33���
�0���ᙚC�P�?�33��z��1G���{C�e                                    BxaQb  "          @���?��\���ÿ���(z�C�xR?��\�����K��z�C��                                    BxaQ-  �          @�Q�?�����33�u�\)C��)?������
��\���C�7
                                    BxaQ;�  
�          @��?����33��ff���RC��H?�����R�(����G�C�y�                                    BxaQJT  �          @��R?L����z�>u@��C�?L�������G����C�K�                                    BxaQX�  �          @�ff?k����>�z�@8Q�C���?k���Q��(�����C���                                    BxaQg�  "          @�p�?E���33>���@��C��?E�������������C�(�                                    BxaQvF  �          @���>�ff��33?�@�Q�C�Q�>�ff����ٙ���  C�k�                                    BxaQ��  �          @�ff?
=���>��@)��C���?
=�����   ��=qC�'�                                    BxaQ��  
Z          @�p�=u����>��@+�C�Q�=u�������R���RC�XR                                    BxaQ�8  T          @��>W
=����>���@C33C�>W
=��녿�����\)C�                                      BxaQ��  �          @�p�>W
=��33?=p�@�  C��>W
=��{���
�{
=C�R                                    BxaQ��  "          @��
>aG����?���A9�C�!H>aG���������9p�C�!H                                    BxaQ�*  
�          @��
?(���\)?fffA�
C�4{?(���zῪ=q�\(�C�AH                                    BxaQ��  �          @�\)?�����
?J=q@��C�?����\)��p��o�C�"�                                    BxaQ�v  
�          @�
=?0�����?fffA��C��?0����Q쿮{�\��C��{                                    BxaQ�  
�          @���?B�\����?:�H@�RC�� ?B�\���Ϳ�  �u�C��)                                    BxaR�  
Z          @��
?8Q�����?333@�z�C���?8Q�����\�z�RC��R                                    BxaRh  
�          @�p�?Y�����\?��@��C�Z�?Y����33������C���                                    BxaR&  '          @�z�?W
=��G�?#�
@У�C�L�?W
=��33������G�C�q�                                    BxaR4�  �          @���?�{����>�
=@�33C��=?�{��z��(�����C�{                                    BxaRCZ  
#          @�G�?�33���\���Ϳ��C�Z�?�33���
�
�H���C��                                    BxaRR   �          @���?�
=���H����Q�C���?�
=����\)��  C�g�                                    BxaR`�  
�          @���?��H���׿J=q�
=C�?��H�����1G���z�C���                                    BxaRoL  "          @��H?�\)��\)�����\(�C�W
?�\)����O\)���C���                                    BxaR}�  �          @�?��\��\)��ff���
C��?��\��(��k��!C��f                                    BxaR��  �          @�=q?�p������\)��ffC�=q?�p��\(���z��B33C�ٚ                                    BxaR�>  �          @�Q�?�z���녿�  �,(�C���?�z������7����C��                                    BxaR��  �          @�z�?�G����?fffA=qC�ٚ?�G���G����R�J�\C���                                    BxaR��  T          @�z�?�����?J=qA��C�/\?����������\��C�T{                                    BxaR�0  �          @�?������R?:�H@�p�C��?������\���g\)C�>�                                    BxaR��  
�          @�?����?\(�A(�C�/\?����33���
�Q�C�J=                                    BxaR�|  �          @��
?\��z�>�ff@��HC�� ?\���Ϳ�33��ffC�8R                                    BxaR�"  
�          @�p�?�  ���
?uA��C���?�  ���H��z��=p�C��                                     BxaS�  T          @�?�p����
?���A2�HC��R?�p���(����
�'33C��{                                    BxaSn  �          @���?�ff��33>�  @$z�C�P�?�ff��G���ff����C��
                                    BxaS  
Z          @�(�?ٙ����>8Q�?�C��?ٙ����ÿ�\)����C�Z�                                    BxaS-�  T          @���?�{���>��@*=qC�K�?�{��������C�                                    BxaS<`  T          @�33?�ff��(�>�?��C�?�ff���ÿ���=qC���                                    BxaSK  T          @���?�G����\>��?���C��?�G���������ffC�ff                                    BxaSY�  T          @�(�@G���������Y��C���@G���\)��\��(�C��=                                    BxaShR  "          @�(�?�Q���Q쾏\)�:=qC�%?�Q���Q��  ����C��                                    BxaSv�  
�          @��H?����Q�    �#�
C��R?�����
��p���=qC���                                    BxaS��  
�          @���?�\��
=>��@/\)C�` ?�\����(���33C�޸                                    BxaS�D            @��?����(�?���A]C��\?����  �E�����C�q�                                    BxaS��  '          @���?�=q���?\(�A	�C��
?�=q��
=���
�M��C��\                                    BxaS��  
Z          @��\?�����(�?+�@ӅC��)?�����
=���R�l(�C��                                    BxaS�6  "          @�(�?��
����?�@�G�C��
?��
��{�У���
=C��                                     BxaS��  O          @�p�?�����p�?
=q@��C��?������R��{�}G�C�+�                                    BxaS݂  '          @�{?�{��?\)@�p�C��?�{��\)�����yC�1�                                    BxaS�(  "          @�\)?����R?��@�Q�C�(�?���  ��{�z�HC�s3                                    BxaS��  T          @��?�{��  >���@s�
C��?�{�����\��\)C�(�                                    BxaT	t  T          @��R@Q���\)?O\)A{C��@Q������Q��@��C�7
                                    BxaT  "          @��@"�\��?Q�A�RC�� @"�\��(�����.�RC��)                                    BxaT&�  T          @�=q@*�H���\?!G�@�Q�C�}q@*�H��
=�����G\)C��                                    BxaT5f  �          @�G�@7
=��ff>�@���C���@7
=��G���ff�ZffC��                                    BxaTD  �          @���@7
=��
==u?!G�C��R@7
=�����
=���C�w
                                    BxaTR�  "          @�=q@N�R��Q쾳33�hQ�C���@N�R��=q���R��p�C�7
                                    BxaTaX  
�          @��\@dz����þ�{�`  C��=@dz��vff�����{C�e                                    BxaTo�  T          @�33@hQ������{�^{C�G�@hQ��tz��\)��(�C���                                    BxaT~�  T          @��@fff���þ����G�C��@fff�w�������=qC�o\                                    BxaT�J  
�          @�z�@qG����8Q��ffC�@qG��u���Q����
C�G�                                    BxaT��  �          @�{@x�����;�=q�+�C���@x���p�׿�G���{C��{                                    BxaT��  "          @�(�@h����  �
=���C�O\@h���o\)����RC�q                                    BxaT�<  �          @��
@c33��G��(���ٙ�C��3@c33�p��������C��
                                    BxaT��  
�          @���@a�������
�N�RC�� @a��|�Ϳ�\)��  C��                                    BxaTֈ  �          @�{@g���(������HC���@g���G��ٙ����C���                                    BxaT�.  	�          @���@W���Q쾅��)��C�b�@W����
��\)��G�C���                                    BxaT��  
-          @��
@L(����
�W
=�(�C�S3@L(���\)��{��33C��                                    BxaUz  "          @��@Z�H�����R�H��C���@Z�H���׿������C�5�                                    BxaU   �          @��@X�����R�Ǯ��Q�C��)@X�����׿�(���=qC�\                                    BxaU�  �          @�(�@P�����H����  C���@P�������G����
C�                                    BxaU.l  �          @���@L(����;����@  C�@ @L(���\)��Q�����C���                                    BxaU=  "          @�{@N�R����   ����C�\)@N�R���Q����C��q                                    BxaUK�  �          @�p�@P  ���
���H���C��@P  ��z��ff��{C��                                    BxaUZ^  "          @�(�@R�\��G�����C���@R�\��=q�z����\C�xR                                    BxaUi  
�          @�p�@J=q��{�����E�C��R@J=q���ÿ�����G�C�8R                                    BxaUw�  
�          @���@<�����=L��>�C�� @<����Q��z���G�C��3                                    BxaU�P  �          @��R@g����\�@  ��G�C��{@g��q���R����C�޸                                    BxaU��  
�          @�{@U��=q�����C�)@U�����\���\C���                                    BxaU��  T          @��@h�����H�@  ���C��@h���r�\��R��{C��=                                    BxaU�B  �          @��@J�H��{=�\)?=p�C�
=@J�H�����=q���
C�ٚ                                    BxaU��  
�          @��R@^{���ýL�;��C���@^{��
=�����G�C��{                                    BxaUώ  �          @�\)@z=q��p������}p�C���@z=q�p�׿�=q��G�C��                                    BxaU�4  
�          @�\)@O\)��>.{?��HC�\)@O\)��{��p��p��C��                                    BxaU��  
�          @�p�@!G���G�>�
=@�\)C�G�@!G����
��\)�`��C��                                    BxaU��  
�          @�
=@$z����?(��@��C�}q@$z���
=����8��C���                                    BxaV
&  
Z          @��@/\)���R?k�A(�C�t{@/\)��
=�\(��
{C�l�                                    BxaV�            @��@��L��?�(�A�z�C�G�@��fff?   @��C���                                    BxaV'r  
_          @�@:=q��Q�@�\A�ffC�,�@:=q��ff>�(�@�\)C��H                                    BxaV6  
�          @�=q?�Q�����?�=qA�C���?�Q��������Q�C�%                                    BxaVD�  
�          @�@N�R���@   A�G�C�3@N�R����>�ff@��C���                                    BxaVSd  �          @��@ ���aG�@AG�B��C��@ ����33?�\)A�p�C�f                                    BxaVb
  �          @�{@z��w
=@(�A㙚C���@z���
=?uA-p�C��q                                    BxaVp�  
�          @��@#�
��  ?��RA�=qC�L�@#�
��{?   @��C��R                                    BxaVV  "          @�z�@&ff��z�?���A���C��@&ff��p�=u?0��C�5�                                    BxaV��  �          @�\)@"�\��Q�?��A�(�C�k�@"�\���=��
?fffC���                                    BxaV��  
�          @��@,���mp�@�\A�(�C��@,����Q�?aG�A
=C�&f                                    BxaV�H  �          @��@%�`  @*=qA�G�C�C�@%��{?��Am��C���                                    BxaV��  T          @�\)@�
���R@�AǙ�C��@�
��ff?�R@�z�C��{                                    BxaVȔ  "          @���@!��\)@(�A�
=C�33@!���  ?5@��C���                                    BxaV�:  	�          @���@O\)�W
=@(�A܏\C�  @O\)�~�R?�z�AJ�HC���                                    BxaV��  T          @��R@aG��L��@�A�p�C�� @aG��mp�?aG�A��C���                                    BxaV�  "          @�Q�@!��L��@J=qB
=C�"�@!����\?�33A�ffC��                                    BxaW,  
�          @��@\)�1�@g
=B5(�C�g�@\)�u�@p�A��C�"�                                    BxaW�  "          @��@8���@  @:�HB�C��@8���s�
?޸RA�C��{                                    BxaW x  T          @��R@ff��
=?z�@�z�C��\@ff��p��\(��#�C��3                                    BxaW/  
�          @��H@
�H�c33?n{AI�C�@
�H�i�����R����C�aH                                    BxaW=�  �          @�
=@5��W
=@w
=BR�\C��f@5���@W
=B/(�C��)                                    BxaWLj  
�          @�33@��?L��@�  Bz33A���@�Ϳ@  @�Q�B{
=C��                                     BxaW[  "          @�@ff�h��@��RBlG�C�e@ff���@j�HBAffC��                                    BxaWi�  �          @�(�@녿�G�@���B_
=C�B�@��0��@S33B*p�C��\                                    BxaWx\  T          @���@�
��{@�=qBbz�C���@�
�(Q�@X��B0=qC��H                                    BxaW�  �          @�p�@
=��@s33BJ�HC��q@
=�J�H@:�HB��C�K�                                    BxaW��  
�          @�p�@���"�\@]p�B5�C��R@���b�\@�A�Q�C�L�                                    BxaW�N  "          @�\)?��R�Dz�@>�RB�C�t{?��R�xQ�?�A�ffC��                                    BxaW��  T          @�
=@��7�@FffB$��C���@��n�R?��RA��C���                                    BxaW��  "          @���@��\)@W
=B3=qC�C�@��\��@
=A�G�C��                                     BxaW�@  "          @�
=@*=q��{@p  BD\)C���@*=q�>�R@<(�B33C��                                    BxaW��  
�          @�\)@�ٙ�@�G�BY�\C��
@�;�@QG�B$=qC�O\                                    BxaW�  
�          @��?�?��@�p�B�ǮB  ?���(�@���B�L�C��)                                    BxaW�2  
�          @���?�  @   @��Bfz�B\(�?�  ?L��@��B��Ař�                                    BxaX
�  
�          @�p�?�z�@X��@�(�BK  B�Q�?�z�?��@��B���Bdp�                                    BxaX~  
�          @�=q?}p�@hQ�@��
BD�
B�aH?}p�@�@��B��HB�                                    BxaX($  T          @�  ?���@g
=@�{B;33B���?���@�
@�(�B|�BS                                      BxaX6�  �          @��
?���@k�@�B5�HB~��?���@��@�z�Bu33BFff                                    BxaXEp  �          @���?�\)@|��@�G�B-��B�L�?�\)@(�@��HBp33Ba��                                    BxaXT  �          @���?��@�p�@p  B'
=B��f?��@0  @�z�BpB�=q                                    BxaXb�  "          @��?+�@��\@P��B33B�G�?+�@a�@�z�BP��B���                                    BxaXqb  
�          @�(�?u@�\)@<(�A�z�B��?u@r�\@��
B@�
B�u�                                    BxaX�  
�          @��>\@�{@dz�B�RB���>\@S�
@��
B^�B�Ǯ                                    BxaX��  "          @��n{@ ��@��Bw�Bܨ��n{?L��@���B�  C\)                                    BxaX�T  "          @��u@1�@��\Bj�B��ÿu?�{@�
=B���CǮ                                    BxaX��  
�          @��\�#�
@K�@���BY33B��Ϳ#�
?˅@�G�B�
=B��f                                    BxaX��  
�          @�    @p  @z�HB9  B�      @�@���B�8RB���                                    BxaX�F  �          @�녾��R@�Q�@uB.�
B��;��R@&ff@���BwB��=                                    BxaX��  T          @��׿#�
@�\)@^{B��B���#�
@;�@��
Bc�B̀                                     BxaX�  "          @�p���=q@��@`��B!�B�p���=q@3�
@��
Bj=qB��)                                    BxaX�8  �          @��
?u@|��@q�B+��B��{?u@$z�@�=qBqG�B���                                    BxaY�  	�          @��
?z�H@x��@z=qB1{B�Ǯ?z�H@�R@�Bv\)B��                                    BxaY�  O          @���?J=q@�ff@aG�B�HB���?J=q@9��@�z�Bd�B�u�                                    BxaY!*  '          @�=q?�z�@��\@HQ�B��B��)?�z�@:=q@��BR�B�G�                                    BxaY/�  	�          @�{?�33@�z�@z�AծB��?�33@\��@c�
B.�RB���                                    BxaY>v  
(          @��?Q�@�33@�A�(�B��)?Q�@o\)@Y��B&��B�Q�                                    BxaYM  "          @�33>�z�@��?�p�Ae�B���>�z�@�ff@*=qB
=B���                                    BxaY[�  �          @���B�\@�G�?�ffAr=qB�k��B�\@��@-p�B  B�{                                    BxaYjh  
�          @���?�{@���@(Q�A�B��?�{@`  @x��B:��B��                                    BxaYy  
(          @�  ?#�
@�33@(�A�=qB�?#�
@w�@s�
B0�\B�L�                                    BxaY��  �          @���?(�@���@��A�(�B�\?(�@��@hQ�B$p�B�                                    BxaY�Z  
�          @�{�#�
@�z�?޸RA���B�33�#�
@��
@N{B��B�8R                                    BxaY�   "          @���>���@���?���A��
B�L�>���@�@\��BQ�B���                                    BxaY��  
�          @��=u@��
?��A�  B��q=u@��@P��Bp�B��\                                    BxaY�L  T          @�
=<#�
@���?�(�Ad(�B���<#�
@��@H��A��\B���                                    BxaY��  �          @��#�
@�ff?˅AyB���#�
@�ff@N�RB�B�#�                                    BxaYߘ  �          @�=q>�@��?�  Ao�B�Q�>�@���@G�BB��                                    BxaY�>  T          @�{���
@�(�@�A�ffB��
���
@}p�@l(�B+��B�G�                                    BxaY��  
�          @��\��
=@��
@ffA�ffB���
=@���@]p�B"p�B�                                    BxaZ�  �          @�33���@��R?�Q�A�B�녾��@��@U�B�B�                                      BxaZ0  �          @�p����@�G�?��HA���B�Q���@�\)@W�BB��                                    BxaZ(�  T          @�G���=q@�(�@�
A�z�B�
=��=q@�G�@^�RB(�B�8R                                    BxaZ7|  �          @�논#�
@��H@�A���B�(��#�
@�@p  B!33B�.                                    BxaZF"  �          @��>\)@��
@(�AîB���>\)@��@z�HB&z�B�Q�                                    BxaZT�  �          @�G�=L��@�33@��A�(�B��=L��@�  @c33B��B��R                                    BxaZcn  T          @�=L��@��\@��AׅB��R=L��@y��@p��B0
=B�k�                                    BxaZr  �          @���#�
@�Q�@��AٮB���#�
@u@o\)B0�B��q                                    BxaZ��  T          @��
�8Q�@��H@33A�\)B�Q�8Q�@}p�@g�B)�\B�G�                                    BxaZ�`  �          @�z�<�@�Q�@!G�A���B�8R<�@tz�@r�\B3=qB�                                    BxaZ�  �          @������@�{@��A�=qB�Lͽ���@���@n{B*�RB���                                    BxaZ��  
�          @�����
@�33@��A��B�uü��
@|(�@mp�B-  B��{                                    BxaZ�R  �          @���#�
@�G�@�HA�Q�B���#�
@xQ�@l��B.ffB�8R                                    BxaZ��  
�          @�p�>��@��R@*�HA�B���>��@o\)@z=qB8��B�
=                                    BxaZ؞  �          @�>Ǯ@���@!�A�  B��{>Ǯ@w
=@r�\B1��B�W
                                    BxaZ�D            @�=#�
@���@2�\A�\)B���=#�
@i��@�  B>ffB��3                                    BxaZ��  
�          @��
��\)@��H@1�A�B��R��\)@fff@~�RB?ffB�33                                    Bxa[�  �          @�z�\)@�@(��A�p�B�p��\)@n�R@w�B8  B�Q�                                    Bxa[6  T          @��>#�
@��\@7�B =qB�{>#�
@dz�@���BB\)B���                                    Bxa[!�  	�          @�ff=���@��H@;�BG�B��=���@dz�@��
BDG�B���                                    Bxa[0�  
�          @�ff>��
@�ff@.�RA��\B��>��
@o\)@|��B9�B�\                                    Bxa[?(  T          @�G�?��@�  @1G�A���B��q?��@qG�@�  B9z�B�W
                                    Bxa[M�  �          @�Q�?(�@��R@1G�A�B�B�?(�@o\)@~�RB9�B��                                     Bxa[\t  "          @��>�=q@��@$z�A㙚B��>�=q@tz�@s33B2��B���                                    Bxa[k  
�          @������@��\@�A�=qB�.����@\)@c33B&{B���                                    Bxa[y�  
Z          @�\)>aG�@��H@!�A܏\B��3>aG�@|(�@q�B/�B�z�                                    Bxa[�f  �          @�G���33@��@�AǮB��ᾳ33@�z�@g�B$ffB���                                    Bxa[�  "          @����Q�@�=q@33A�p�B����Q�@\)@c33B&{B�G�                                    Bxa[��  "          @�\)����@���?���A�\)Bޅ����@�=q@FffB{B�
=                                    Bxa[�X  "          @��H��33@��
?���A�  Bҙ���33@��@;�A�G�B֮                                    Bxa[��  
�          @�z�k�@��@p�A�\)B�
=�k�@�33@c33B��B���                                    Bxa[Ѥ  
Z          @��;W
=@�=q?�A�{B���W
=@��@N�RB�
B�8R                                    Bxa[�J  
�          @��
��{@��?�\)A���B�L;�{@��R@A�B�HB�Q�                                    Bxa[��  T          @�=q���@��\?�(�Au�B��
���@�
=@7�A�33B�\)                                    Bxa[��  T          @���   @�  ?�p�A���B��H�   @�Q�@UB\)B���                                    Bxa\<  T          @�p����@�@,��A噚B�녾��@�Q�@|(�B1�HB�G�                                    Bxa\�  �          @���   @�  @�A���B�\)�   @�  @VffBffB�\)                                    Bxa\)�  �          @���G�@�?�G�A%�B��῁G�@��R@(�A�=qB��                                    Bxa\8.  �          @��H����@��׿fff�  Bυ����@��?�R@ƸRB�W
                                    Bxa\F�  T          @���Tz�@������N{Bģ׿Tz�@���>��@$z�B�=q                                    Bxa\Uz  �          @�G���33@�Q�    =L��B�𤾳33@���?�ffAyG�B�33                                    Bxa\d   �          @�z�@  @���>k�@��B�\)�@  @���?ٙ�A�(�B��                                    Bxa\r�  �          @�z��\@���?^�RA33B�z��\@��H@z�A��HB�\)                                    Bxa\�l  �          @�G�=�G�@��
?���AW\)B�=�G�@��@2�\A��B��                                     Bxa\�  T          @�=q?   @��@EB
=B��{?   @dz�@�{BE�B�.                                    Bxa\��  T          @�Q�>�\)@��@>�RA��B��{>�\)@|��@�p�B:
=B�                                    Bxa\�^  T          @��?�\@���@^{B�B�.?�\@]p�@�G�BQ��B�8R                                    Bxa\�  
�          @���=�Q�@�z�@�G�B1=qB��{=�Q�@;�@��Bn=qB��{                                    Bxa\ʪ  �          @��H>�{@n�R@�\)BH\)B�� >�{@�H@���B�k�B��f                                    Bxa\�P  "          @��H>��H@c�
@�33BP  B�p�>��H@�R@��
B��fB�ff                                    Bxa\��  "          @�G�>��
@G
=@��Bd�B�8R>��
?޸R@�  B�p�B�
=                                    Bxa\��  �          @�G�>�  @I��@��Bc�B���>�  ?�\@�Q�B���B�                                    Bxa]B  T          @�\)>.{@dz�@���BUG�B��=>.{@��@��B���B�(�                                    Bxa]�  T          @��>8Q�@X��@�{B]�
B��
>8Q�@   @�z�B��
B��{                                    Bxa]"�  "          @�{>.{@`  @���BWG�B�u�>.{@	��@�G�B�z�B���                                    Bxa]14  T          @��R>���@J=q@���Bg{B�Q�>���?�G�@�p�B��B��                                    Bxa]?�  
�          @�33>�G�@(��@�  B{�HB�G�>�G�?��H@�  B��
B�W
                                    Bxa]N�  
�          @�33?&ff@��������=p�B�#�?&ff@�{?�Q�A<Q�B��H                                    Bxa]]&  
�          @��>�ff@��>�=q@(Q�B�.>�ff@���?޸RA��HB�                                    Bxa]k�  
Z          @�>�33@�33?h��A��B��>�33@�p�@
=A�  B��{                                    Bxa]zr  �          @��?�z�@�33?�\)A�(�B�k�?�z�@���@:�HA�\)B���                                    Bxa]�  �          @�z�?�p�@�(�@'�A��
B��?�p�@��@qG�B�B�{                                    Bxa]��  
�          @���?�{@���@@  A�B��?�{@�33@��B/B�Q�                                    Bxa]�d  
�          @�{?�Q�@��
@_\)BB�
=?�Q�@fff@�G�BH�B�B�                                    Bxa]�
  �          @���?޸R@Mp�@�
=BKG�Bv\)?޸R?�p�@�z�B{Q�BCG�                                    Bxa]ð  
�          @�
=@�
?���@�  Bt�RB%p�@�
?�\@�G�B�8RA^=q                                    Bxa]�V  T          @��H@z�@ff@�
=Bl(�BB=q@z�?xQ�@���B���A�33                                    Bxa]��  �          @�@33@@���Bm��B5��@33?c�
@��RB�A���                                    Bxa]�  "          @�Q�@z�?�ff@�(�B���B�@z�>.{@�33B�W
@�                                      Bxa]�H  �          @�\)@zᾣ�
@�  B���C��@z��p�@�  Bx�
C���                                    Bxa^�  �          @��@��>�@�p�B���@E@�Ϳ�=q@���B�k�C���                                    Bxa^�  T          @�z�@�?(��@�
=B�B�A�(�@��Tz�@�ffB�.C��
                                    Bxa^*:  �          @�p�@��?E�@�ffB�(�A���@�Ϳ5@�ffB��{C�)                                    Bxa^8�  �          @��@ff?�{@�Q�B�z�A�@ff����@�33B��fC���                                    Bxa^G�  
�          @�Q�@�?^�R@�ffB�33A�@��
=q@��B�\C��)                                    Bxa^V,  "          @�Q�@��?=p�@��RB���A��R@�ÿ+�@�
=B�C�'�                                    Bxa^d�  T          @�=q@�?��@�33B�=qAZ�R@녿aG�@��B�\)C�~�                                    Bxa^sx  �          @�Q�@{>�z�@�  B�.@׮@{����@���B���C���                                    Bxa^�  �          @��@Q�?L��@��B��
A�
=@Q��@�z�B�#�C�=q                                    Bxa^��  �          @�
=@	��?�(�@�z�B�A�ff@	����G�@���B�p�C��=                                    Bxa^�j  T          @�(�@z�?�(�@��RB}{B�@z�=�@���B�G�@E�                                    Bxa^�  �          @�
=@��?�ff@���Bz�Bp�@��>B�\@�\)B�k�@���                                    Bxa^��  �          @ƸR@��@33@��Bf��B/G�@��?}p�@�z�B���A��                                    Bxa^�\  �          @�
=@  @+�@�z�B^�BG�R@  ?��@�(�B���A��R                                    Bxa^�  
�          @��H@�@\)@�G�Bb��B6ff@�?�@�\)B��A���                                    Bxa^�  
�          @Ϯ@%?�@��B��A7�
@%�\(�@���B�k�C��R                                    Bxa^�N  
�          @�{@
=���
@�  B�8RC��@
=���
@�  B~��C�z�                                    Bxa_�  
�          @�(�@�þu@�B�z�C�'�@�ÿ�
=@�ffBffC�z�                                    Bxa_�  �          @�@녾�  @���B�Q�C��R@녿�33@�G�B�8RC��                                    Bxa_#@  �          @�z�@�>�{@�p�B�  AG�@���ff@��\B��C��                                    Bxa_1�  T          @�G�?��R�L��@��RB�#�C�G�?��R��
=@�G�B�#�C�)                                    Bxa_@�  T          @�@��>��@�G�B��AE@�ÿp��@��B�Q�C�#�                                    Bxa_O2  T          @��?��?�=q@�=qB�8RB�\?�녽���@ƸRB�
=C�u�                                    Bxa_]�  
�          @��?�p�?���@��
B���BA?�p�>��H@��B��RA|Q�                                    Bxa_l~  �          @�ff?��@�
@\B���B>ff?��?(�@�z�B��A�p�                                    Bxa_{$  T          @�{?�\@	��@��HB��BJ(�?�\?333@�p�B�
=A�z�                                    Bxa_��  �          @ٙ�?��?�33@�  B���B4z�?��>�
=@У�B��AI��                                    Bxa_�p  �          @�G�?��R?��@ə�B�p�Bz�?��R=�\)@ϮB���?�33                                    Bxa_�  �          @�{?�p�?��H@ȣ�B�G�A�?�p��u@�(�B�=qC���                                    Bxa_��  �          @�(�?�p�?���@ƸRB�
=A��\?�p��u@�=qB��HC���                                    Bxa_�b  T          @θR?�?!G�@�z�B��
A�ff?��B�\@��
B�  C���                                    Bxa_�  �          @�p�?�  ?E�@��
B�\)A��R?�  ��R@�z�B�ffC�7
                                    Bxa_�  T          @��?�z�>�G�@��B�aHAm�?�z�n{@�  B��HC�ff                                    Bxa_�T  T          @�?�33>�@�=qB�B�A`��?�33�c�
@���B���C�s3                                    Bxa_��  T          @�ff@�?z�@\B���A�z�@녿G�@��B�� C�|)                                    Bxa`�  
�          @�\)@�>�@�z�B�ffAU�@녿c�
@��HB���C�.                                    Bxa`F  �          @��@�
?xQ�@�=qB�#�A�G�@�
�Ǯ@�(�B�=qC�7
                                    Bxa`*�  T          @�(�@&ff?�=q@��RB�A�z�@&ff���
@ÅB��C��                                    Bxa`9�  T          @��
@C33?�Q�@��Bh�B�@C33?�R@��HB{�A8Q�                                    Bxa`H8  �          @�z�@L��@
=q@��B_
=B{@L��?^�R@�  Btp�Arff                                    Bxa`V�  �          @��@*�H?�@�{Bq��Bff@*�H?�@�{B��RA2ff                                    Bxa`e�  
�          @��H@��?aG�@�Q�B�8RA��@���Ǯ@�=qB��3C�Z�                                    Bxa`t*  �          @ʏ\?��R�aG�@�  B�.C��
?��R�\@��B�z�C�S3                                    Bxa`��  
�          @�ff@녾��@�=qB��=C�o\@녿�(�@��\B���C��H                                    Bxa`�v  
�          @ʏ\?�{��ff@�ffB�u�C�7
?�{��
@��\Bx
=C�h�                                    Bxa`�  T          @���?�Q�z�H@�  B��{C��H?�Q��(�@���BvQ�C�Ǯ                                    Bxa`��  �          @�
=@녿��\@�G�B�(�C���@��\)@�{Bs��C�q                                    Bxa`�h  "          @�Q�?�33��=q@�Q�B��=C��3?�33�>{@��Bb�RC��                                    Bxa`�  �          @���?k��B�\@��HBh33C�h�?k�����@��\B8�C�b�                                    Bxa`ڴ  �          @��H?���
�H@��HB�z�C�˅?���S33@�Q�B]G�C��R                                    Bxa`�Z  T          @�(�?�p����R@�
=B�(�C�>�?�p���H@��\BsG�C���                                    Bxa`�   T          @�p�@��?�Q�@�=qB�W
B�@��>k�@��B��@���                                    Bxaa�  �          @ƸR@B�\@��@��\BL�HBG�@B�\?��@�
=Bf(�A�                                    BxaaL  �          @�\)@W�@333@�p�B4�HB�H@W�?�@��BP�A�Q�                                    Bxaa#�  �          @�Q�@I��@.{@�z�B@�\B#z�@I��?�Q�@��B\(�A�=q                                    Bxaa2�  �          @�z�@L��@5�@�\)B?�\B%�
@L��?��
@��RB[�\A��H                                    BxaaA>  �          @θR@G�@AG�@�  B>=qB033@G�?�(�@���B\
=B ��                                    BxaaO�  �          @У�@H��@hQ�@���B*  BD��@H��@(��@��BK�B \)                                    Bxaa^�  �          @�ff@+�@�Q�@Y��A�G�Brp�@+�@}p�@��B((�B_z�                                    Bxaam0  �          @�  @8��@��@X��A�
=Bj�@8��@|(�@�G�B%
=BW
=                                    Bxaa{�  �          @�\)@;�@�z�@\(�B G�Bg�@;�@u@�=qB'33BR                                    Bxaa�|  �          @Ϯ@@  @���@XQ�A�p�Bd�@@  @w
=@�Q�B$=qBP��                                    Bxaa�"  �          @��H@<(�@��@��A��Bs�@<(�@��R@Z�HA��RBg��                                    Bxaa��  �          @��@333@��H@9��A�(�Bt�
@333@�z�@w�Bp�Bf�                                    Bxaa�n  �          @�{@J�H@O\)@��B4��B6�R@J�H@  @�(�BS\)B�R                                    Bxaa�  
Z          @���@[�@C33@��B/��B&��@[�@ff@�{BK�A��                                    BxaaӺ  
(          @�33@u�@HQ�@�33B%(�B(�@u�@��@�z�B?��A��                                    Bxaa�`  
Z          @�33@HQ�@z�@��BV�B  @HQ�?��H@�p�Bmp�A��\                                    Bxaa�  �          @ҏ\@X��@\)@�p�BOQ�B
=@X��?�z�@�Q�Bc�A��
                                    Bxaa��  �          @Ϯ@Fff@{@��\BP=qB{@Fff?�z�@�
=Bhp�A�G�                                    BxabR  T          @�z�@7�@<(�@�(�BG�B6�H@7�?�@��
Be  B�                                    Bxab�  �          @��@[�@-p�@�  B7�BQ�@[�?�\@�{BP  A�Q�                                    Bxab+�  �          @�Q�@(Q�?�@�z�BmQ�B
�@(Q�?5@�z�B�aHArff                                    Bxab:D  "          @ʏ\@A�@�H@���BK��B�@A�?�
=@��Bc�
A��
                                    BxabH�  "          @љ�@�@aG�@fffBp�B �@�@0  @�
=B ��B\)                                    BxabW�  
�          @�(�@��H@dz�@b�\B �
B��@��H@3�
@�p�B�\B�                                    Bxabf6  
�          @�G�@|(�@HQ�@��
B\)B�R@|(�@��@���B6�RA�
=                                    Bxabt�  �          @�\)@x��@:�H@���B&{B��@x��@�@�G�B>�A���                                    Bxab��  	�          @˅@mp�@'�@��RB1��B�
@mp�?��H@�(�BH=qAř�                                    Bxab�(  	�          @��@N�R@z�@�z�BK\)B�H@N�R?��@��Ba(�A��
                                    Bxab��  B          @�z�@3�
@G�@��\B^��B{@3�
?�G�@��
Bs�HA�                                      Bxab�t   d          @���@#33@��@���B]
=B,�@#33?���@��Bw33A�{                                    Bxab�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxab��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxab�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxab�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxab��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxacX              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac3J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxacA�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxacP�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac_<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxacm�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac|�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac�               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxac�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad ^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad,P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad:�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxadI�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxadXB              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxadf�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxadu�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxad�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxae
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxae�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxae%V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxae3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxaeB�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxaeQH              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxae_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxaen�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxae}:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxae��   2          @�z�?�{�N�R@��HBE�
C�u�?�{�vff@`��B"�C���                                    Bxae��  �          @�\)?����g�@uB2  C��=?�����ff@L(�B��C��f                                    Bxae�,  
Z          @�(�?k���Q�@I��Bz�C�3?k����R@=qA�G�C��                                     Bxae��  
�          @�G�?k����R@Dz�B�RC�"�?k����@A��C���                                    Bxae�x  �          @��R>�����H@J=qB
=C��q>������@��AᙚC��3                                    Bxae�  �          @�p�>��H���
@C�
B��C�b�>��H���@ffA�
=C�\                                    Bxae��  
Z          @��?
=q�^�R@p  B;
=C�k�?
=q��G�@H��B=qC�Ф                                    Bxae�j  �          @��R?&ff�`��@��BD=qC�5�?&ff��(�@`  B�C�q�                                    Bxaf  �          @�p�>�Q��|(�@l��B,=qC���>�Q����@AG�BffC�J=                                    Bxaf�  	�          @�ff>\���R@[�B�C��3>\��ff@-p�A���C�N                                    Bxaf\  "          @��
>\��  @9��BC�o\>\���@��A�=qC�<)                                    Bxaf-  "          @�33?.{�XQ�@Q�B.p�C��=?.{�w
=@,��B
ffC���                                    Bxaf;�  �          @�Q�?�녿�  @��Bu\)C�=q?�녿�33@y��B^{C�w
                                    BxafJN  �          @���@�ÿ��@�\)B~p�C�+�@�ÿ�ff@��\BoC�XR                                    BxafX�  T          @���@(��L��@���By(�C�k�@(��G�@��HBs�C�)                                    Bxafg�  T          @��@�=��
@��By��?��H@��&ff@��
BuC�z�                                    Bxafv@  
�          @�(�?���?(�@���B���A�?��ͽ�G�@��\B��=C��                                    Bxaf��  "          @���@�?�33@���Bs�A�@�>��@���B�=qAK33                                    Bxaf��  T          @��@(��?fff@�=qBdz�A��
@(��>W
=@��Bk��@�=q                                    Bxaf�2  �          @�G�@?\)?��
@�
=BTz�A�(�@?\)>�Q�@��\B\�\@���                                    Bxaf��  
�          @��
@333>�G�@��Bh��A\)@333��=q@�=qBi�C�8R                                    Bxaf�~  �          @�=q@녽�G�@�G�B��C��
@녿Y��@�
=Bz�HC��)                                    Bxaf�$  
Z          @�z�@\)���@��B�
C���@\)�xQ�@���Bv��C�XR                                    Bxaf��  T          @�Q�@���\)@���B�p�C�1�@��z�H@�Bx�HC��                                     Bxaf�p  T          @�  @���33@�Q�B���C�B�@����@���Bw  C��                                    Bxaf�  �          @�G�@(���R@�\)B|  C��@(���ff@��\Bm�C���                                    Bxag�  �          @�33@(����\@�\)Bv��C���@(���Q�@���BdffC�#�                                    Bxagb  �          @��@��c�
@��
BC�q�@���=q@�{Bm��C�`                                     Bxag&  �          @�ff@�����@�Bj=qC�o\@���p�@�BVQ�C��                                    Bxag4�  �          @�p�?�녿��@�p�Bk=qC�o\?���"�\@��\BP�RC�J=                                    BxagCT  
�          @��R@p��\)@�\)Brz�C���@p���p�@�33Bf�RC��H                                    BxagQ�  "          @��@8��?\)@�z�BZz�A0(�@8�ý�\)@�B]ffC�P�                                    Bxag`�  
�          @�33@*�H��ff@��Bn=qC�:�@*�H���@��Bdp�C�y�                                    BxagoF  
(          @���@8Q�>���@�z�Ba��@��@8Q쾅�@���Bb�C�q�                                    Bxag}�  
�          @�z�@E?�{@�(�BG(�A�z�@E?z�H@�=qBTp�A�z�                                    Bxag��  �          @���@2�\?���@�  BO�RB�@2�\?�@�\)B_�HA�p�                                    Bxag�8  T          @��@2�\?�33@�33BYQ�A���@2�\?=p�@�Q�Bez�Alz�                                    Bxag��  
�          @�33@E?��\@��BQ��A��
@E>\@��HBYp�@޸R                                    Bxag��  T          @�  @(�ÿ+�@�Bc�C���@(�ÿ��@�G�BWC�
=                                    Bxag�*  T          @�Q�?�(��%�@z�HBN��C��
?�(��HQ�@_\)B1�\C�b�                                    Bxag��  "          @�ff?�G��-p�@vffBLffC��{?�G��P  @Z=qB.33C�|)                                    Bxag�v  �          @�\)?���,(�@y��BN  C��H?���N�R@]p�B0
=C��R                                    Bxag�  "          @�G�?�p��7�@z�HBM��C���?�p��Z=q@\��B.p�C��                                    Bxah�  
�          @���?���C�
@w�BL��C�
?���fff@XQ�B+��C�Y�                                    Bxahh  �          @��
?�
=�C33@w�BG(�C���?�
=�e�@XQ�B'��C�R                                    Bxah  
�          @�p�?����AG�@uBAz�C���?����b�\@W
=B#ffC�#�                                    Bxah-�  T          @�p�@
=q�4z�@o\)B9�
C��q@
=q�U�@R�\B�\C�}q                                    Bxah<Z  T          @�{?���:=q@uB?��C�l�?���[�@W�B#(�C�W
                                    BxahK   
Z          @�\)?�ff�`��@n{B5��C�\)?�ff��Q�@K�B�
C�]q                                    BxahY�  "          @�Q�?��H�XQ�@s�
B:�
C�ٚ?��H�x��@Q�BffC���                                    BxahhL  
�          @�ff?c�
�O\)@|��BF�C���?c�
�qG�@\��B&��C��                                     Bxahv�  
�          @�{?˅�3�
@�=qBNG�C��=?˅�W
=@hQ�B1  C��                                    Bxah��  �          @�@�
�33@�G�B_�RC��q@�
�(��@}p�BG
=C�H                                    Bxah�>  T          @�\)@�\��33@��Bw(�C�� @�\��\@��Bb
=C�u�                                    Bxah��  �          @��@�Ϳ���@�
=Bf
=C���@���z�@�
=BS  C���                                    Bxah��  �          @��H@*�H��G�@�{Ba�C�` @*�H���@�
=BQ  C�q�                                    Bxah�0  
�          @�G�@#�
���\@�Q�Bj�C�/\@#�
��33@�=qB[(�C���                                    Bxah��  �          @�(�?�\)���@�G�B{(�C��{?�\)�G�@���BeQ�C�o\                                    Bxah�|  "          @�@�ÿ.{@���Bu�C��@�ÿ��@�z�Bh�C���                                    Bxah�"  
�          @��R@+����@�Bi{C�8R@+���@���B_=qC�@                                     Bxah��  �          @�ff@.{�#�
@�(�BeC�Z�@.{��  @�  B[33C��f                                    Bxai	n  �          @�{@.�R�L��@��\Bb��C�˅@.�R��33@�BVC�ff                                    Bxai  T          @�ff@)���\@�ffBWC�#�@)���@|(�BE�RC��=                                    Bxai&�  
�          @�@#33��\@�(�BT=qC�� @#33��@uB@(�C��                                    Bxai5`  
�          @��\@'
=��@�p�B`Q�C���@'
=��@�BN�RC�3                                    BxaiD  
�          @�
=@���
@�BYQ�C���@��)��@�33BC�C�9�                                    BxaiR�  "          @���@.�R���@��RBUQ�C��R@.�R�\)@���BA\)C��                                    BxaiaR  �          @���@5���(�@�BT��C�U�@5��z�@���BB\)C�U�                                    Bxaio�  "          @��@.�R��33@���BW33C��3@.�R� ��@�  BCQ�C��3                                    Bxai~�  "          @�G�@)����R@�z�BT�C���@)���6ff@�G�B?{C��H                                    Bxai�D  �          @���@<(���\)@��HBS
=C��H@<(��\)@�G�B@=qC��                                    Bxai��  �          @���@>�R�Ǯ@�
=BZ{C�0�@>�R�(�@��RBIQ�C��{                                    Bxai��  T          @�(�@G�����@�=qB\�
C��H@G�����@��BO{C��{                                    Bxai�6  "          @�=q@J�H��\)@�Q�BN�C�y�@J�H�{@��B>p�C��H                                    Bxai��  �          @�@_\)���H@�Q�BG��C���@_\)��
@�Q�B9�RC���                                    Bxaiւ  �          @��R@QG���\)@���BH��C��@QG��{@�
=B7\)C�}q                                    Bxai�(  
�          @�ff@=p��*=q@��B?�RC�
=@=p��N�R@�Q�B)�RC�=q                                    Bxai��  T          @��@>�R�*=q@��HB=p�C��@>�R�N�R@|(�B'�C�`                                     Bxajt  �          @��R@Z=q�ff@���B7�RC���@Z=q�9��@z�HB$C���                                    Bxaj  �          @�(�@H���*=q@�{B6  C��{@H���Mp�@r�\B ��C�1�                                    Bxaj�  "          @�z�@Y���#33@��HB/�\C��{@Y���Dz�@l��B��C��                                    Bxaj.f  �          @�p�@fff��\@�33B/\)C��@fff�4z�@p  Bz�C��
                                    Bxaj=  
�          @�
=@h����@�ffB4=qC�%@h���(Q�@x��B#z�C�q                                    BxajK�  
Z          @\@n{�$z�@���B&�
C���@n{�E�@h��B33C�*=                                    BxajZX  "          @\@|���#�
@vffB�C��@|���C33@^{B�C�&f                                    Bxajh�  
�          @�G�@p���1�@s�
B�C���@p���QG�@Y��B	G�C��H                                    Bxajw�  �          @�@Y���>{@uB!z�C�o\@Y���\��@Z=qBp�C�Ff                                    Bxaj�J  �          @��R@U�N�R@n�RB��C��
@U�l��@P��B�C��                                    Bxaj��  T          @�
=@W
=�\��@`��Bz�C��@W
=�xQ�@AG�A��C�k�                                    Bxaj��  �          @�{@e�y��@!G�A�p�C�Q�@e��ff?�p�A�C�C�                                    Bxaj�<  �          @�p�@e��S�
@R�\B��C���@e��n{@4z�A�G�C���                                    Bxaj��  �          @���@`  �2�\@tz�B!��C��R@`  �QG�@Z=qB
=C�xR                                    Bxajψ  
�          @�(�@K��l��@G�Bp�C�XR@K����\@'
=A��C��R                                    Bxaj�.  �          @�Q�@I����G�@1G�AۅC�&f@I�����
@�A�{C�'�                                    Bxaj��  "          @���@R�\��p�@5A���C�%@R�\��  @��A���C�3                                    Bxaj�z  �          @�G�@^�R�u�@E�A�Q�C�"�@^�R��ff@#33AȸRC���                                    Bxak
   
�          @�  @Vff�w
=@I��A�33C�~�@Vff���@'�AθRC�,�                                    Bxak�  �          @��@R�\�y��@Mp�B �C�R@R�\��G�@*�HA�ffC��                                    Bxak'l  T          @�{@^�R�x��@3�
A�z�C��@^�R��
=@�A���C���                                    Bxak6  "          @���@J�H���H@EA���C�޸@J�H���R@!�AƏ\C���                                    BxakD�  
Z          @�z�@>�R��Q�@P��B \)C�}q@>�R��z�@*�HA��HC�U�                                    BxakS^  �          @��H@<(��w
=@l(�B�HC���@<(����@I��A�33C�*=                                    Bxakb  
�          @���@@���}p�@eBG�C���@@������@C33A�{C�5�                                    Bxakp�  �          @ƸR@;���G�@k�B��C��3@;����@HQ�A�{C��\                                    BxakP  
�          @�z�@7��s�
@vffBC�~�@7�����@Tz�Bp�C��                                    Bxak��  
�          @�33@J=q���R@G
=A�
=C�w
@J=q��=q@"�\AĸRC�S3                                    Bxak��  �          @���@S�
����@G
=A�
=C�AH@S�
����@#33A�C��                                    Bxak�B  "          @���@-p��}p�@X��B{C�1�@-p����@6ffA�ffC��                                    Bxak��  
�          @�33@B�\���H@R�\Bp�C�K�@B�\��\)@/\)A�z�C�\                                    BxakȎ  
�          @�33@(����(�@R�\B�C���@(������@,��Aҏ\C�|)                                    Bxak�4  �          @��H@�H�p��@�=qB)Q�C�aH@�H��  @c33B  C���                                    Bxak��  T          @�\)@(����
=@i��B�
C��@(�����@E�A��HC���                                    Bxak�  "          @�33@	����\)@��
B1{C�|)@	������@�G�Bp�C��                                    Bxal&  �          @У�@$z���(�@o\)B
=C���@$z���=q@G�A�  C�s3                                    Bxal�  "          @�\)@������@�B��C�
=@����G�@a�A�C�f                                    Bxal r  �          @��H?У���@o\)B
=qC��H?У����@C33A�(�C�)                                    Bxal/  �          @�\)@�����@Dz�A�
=C��R@�����@�A�{C�q                                    Bxal=�  �          @�\)@ff���@6ffA��HC���@ff��\)@Q�A�  C�3                                    BxalLd  "          @�
=?˅���@G�A�=qC���?˅��33?�G�AW33C�P�                                    Bxal[
  �          @�(�?�\)��ff@.{A�=qC�y�?�\)��  @   A��RC��                                    Bxali�  T          @��?��
���@N�RA�z�C��)?��
��\)@#33A�
=C�H                                    BxalxV  �          @�{?��R��
=@R�\A�C�H?��R���H@#�
A��\C�h�                                    Bxal��  T          @��?����33@2�\A�{C�S3?�����@33A��\C�޸                                    Bxal��  �          @��H@   ����@XQ�A�\)C�` @   ���@+�A��\C��R                                    Bxal�H  �          @�=q?�����z�@L(�A�\C���?�����  @�RA��C�aH                                    Bxal��  �          @���?�ff��{@XQ�A���C��=?�ff���\@,��AŮC���                                    Bxal��  �          @��?��R��\)@y��B{C���?��R��@O\)A��
C�
                                    Bxal�:  �          @�z�?�{����@l(�B��C�<)?�{��ff@@  A���C��
                                    Bxal��  T          @�Q�?�����@���B(Q�C��?����33@xQ�BG�C�G�                                    Bxal�  �          @���?�G���G�@�\)B%�C�8R?�G���G�@hQ�B
=qC�\)                                    Bxal�,  T          @�\)@'���  @C33AۅC�AH@'����H@
=A�G�C���                                    Bxam
�  �          @���?�\)���
@p  B�C�ff?�\)����@C33A֏\C��                                    Bxamx  �          @���?}p����@�ffB$��C��)?}p���Q�@r�\B33C�
                                    Bxam(  T          @ڏ\?�{����@�RA���C�|)?�{�ə�?ٙ�Ak33C�.                                    Bxam6�  T          @�ff@(���ə�?L��@�ffC�Y�@(����33    <��
C�E                                    BxamEj  �          @�=q@:�H��\)@5A���C���@:�H��G�@ffA���C��                                    BxamT  
�          @陚?�׾�33@߮B�p�C���?�׿�p�@�z�B��C�c�                                    Bxamb�  �          @�=q@p��  @���Bz  C���@p��A�@��Bd�HC�~�                                    Bxamq\  T          @�\)@7��\)@���B<33C��@7����
@��B$=qC��                                    Bxam�  �          @�Q�@.{��G�@�B+�
C�xR@.{��(�@�=qB�RC��q                                    Bxam��  
�          @���@���ff@�(�B3�C���@����@�  B{C���                                    Bxam�N  
�          @�@1G����
@��RB9�C��
@1G����@���B!{C�(�                                    Bxam��  T          @�\@]p��QG�@���BD��C�L�@]p��|(�@�=qB0
=C���                                    Bxam��  �          @�z�@U��l(�@��B<��C�\@U����\@�33B&z�C���                                    Bxam�@  �          @�R@h������@��B��C�@h����{@|(�A��HC���                                    Bxam��  
�          @�\)@�
=��  @�ffB�C�b�@�
=����@xQ�A���C���                                    Bxam�  
�          @�\)@Z=q���@s33A��C�
@Z=q���H@EA�z�C�q                                    Bxam�2  �          @�\)@<(�����@AG�A�G�C���@<(���\)@�RA��C�5�                                    Bxan�  �          @��@7����
@QG�A�Q�C���@7���\)@�RA�  C��                                    Bxan~  "          @�Q�@3�
�ə�@:=qA��C���@3�
���
@ffA�ffC�|)                                    Bxan!$  "          @��
@�z��	��@���B!=qC�'�@�z��,��@�\)B�C���                                    Bxan/�  
�          @�@���?=p�@��B �@�Q�@���=�@��RB"��?�33                                    Bxan>p  �          @�=q@�ff��\)@��BP�
C��3@�ff�Tz�@�  BN�C��                                    BxanM  "          A�\@�
=���@�=qB   C��@�
=���
@�  B{C�                                    Bxan[�  �          A�@�{��@��\Aۙ�C��q@�{����@UA�G�C��q                                    Bxanjb  T          A�@�����@QG�A��C��f@�����G�@{A}G�C��R                                    Bxany  
�          A��@�  ���H?��A"�HC��f@�  ��?=p�@��\C�<)                                    Bxan��  "          A��@�z���G�@p�Aup�C���@�z���G�?�\)A ��C��                                    Bxan�T  �          A{@��H��p�@��RA���C�f@��H�θR@{�A��C���                                    Bxan��  �          A�\@�����=q@�\)A�C�  @�����@w�A¸RC��                                    Bxan��  	�          AG�@������@��\A��C��R@������
@O\)A�(�C��H                                    Bxan�F  
Z          AQ�@�z���=q@���A�p�C��@�z����@L��A���C�
                                    Bxan��  T          A��@�z���Q�@��A�(�C���@�z����@}p�AȸRC��=                                    Bxanߒ  
�          A�R@N{��\)@У�B/Q�C�%@N{��  @�
=B�RC��                                     Bxan�8  
�          Aff@]p���G�@�
=B'  C��f@]p��أ�@��BC���                                    Bxan��  �          A@U���ff@�33B4  C�+�@U���\)@��HB��C���                                    Bxao�  �          A�@N�R��{@ǮB'(�C�@N�R��p�@�p�BffC�}q                                    Bxao*  	�          A�\@�����@���B�RC�f@�����G�@�ffA�(�C��H                                    Bxao(�  T          A  @�p��Ӆ@�ffA�p�C��q@�p���@��\A��C��                                    Bxao7v  �          A�@�=q�У�@���B��C��q@�=q���@��A�(�C�Ф                                    BxaoF  "          A�@N�R��\)@�\)B>�C�G�@N�R���@ǮB%�C���                                    BxaoT�  �          A�@Vff����@�G�BI�C�Ǯ@Vff��p�@ӅB0��C���                                    Bxaoch  T          A��@N�R��(�@�33BQ�C��q@N�R��G�@�B8��C���                                    Bxaor  �          A�@S�
��{@�Q�B;Q�C�
@S�
�У�@�  B"
=C�q�                                    Bxao��  
(          A��@U���@�p�B9�C�Ff@U��\)@��B �RC��H                                    Bxao�Z  �          A��@>{��Q�@�BQz�C�P�@>{��p�@��
B833C�P�                                    Bxao�   T          A��@^�R���@�=qB?�C�z�@^�R��ff@��HB&�
C��H                                    Bxao��  �          A�R@~{��\)@ᙚB<(�C��)@~{��=q@��HB$��C��R                                    Bxao�L  T          A=q@7
=���@�p�B7��C��f@7
=��(�@ÅBp�C�N                                    Bxao��  
�          A�@"�\��33@��B33C���@"�\���@��HA噚C��                                    Bxaoؘ  "          A(�@Q���=q@�(�B�C�N@Q���Q�@�ffA��HC��=                                    Bxao�>  "          A��@c33�ڏ\@���B=qC��q@c33��ff@�  A���C��q                                    Bxao��  T          A�@%���H@�=qBffC���@%��\)@��A�{C���                                    Bxap�  "          Ap�@=p�� ��@k�A��
C��@=p��
=@(��A�p�C���                                    Bxap0  T          A=q@G��
�R@G�AEC��@G��?h��@���C��3                                    Bxap!�  "          AQ�?�G���{@\B{C���?�G��=q@�33A��C��                                    Bxap0|  �          A��?�p���ff@У�B&33C���?�p���
=@��B	C�AH                                    Bxap?"  
�          Ap�@�
��(�@�p�B"G�C��)@�
��(�@�
=B��C�,�                                    BxapM�  �          A�R=L���G�@��\A���C�/\=L����@o\)A���C�,�                                    Bxap\n  �          A�R?B�\���
@��B�\C���?B�\���@�33A�\)C���                                    Bxapk  �          A=q?=p���(�@��HBQ�C��H?=p����@��\A���C���                                    Bxapy�  "          A�\?�������@�p�B�C�#�?�����@�p�A��C���                                    Bxap�`  T          A��?�G����@�\)B
=C��q?�G��  @�\)B Q�C�S3                                    Bxap�  �          A  ?�ff�陚@ȣ�B 33C���?�ff� ��@���B��C�u�                                    Bxap��  "          A
=?�ff��(�@�(�B$�C�#�?�ff��(�@�B(�C���                                    Bxap�R  "          A�
?�(���\@�B%��C���?�(����H@�\)B	33C�1�                                    Bxap��  �          A@Mp��n�RA��Bm�C�\)@Mp���Q�@��BV��C�                                      Bxapў  
�          A�\@,(��   A=qB�{C���@,(��g
=A  BxQ�C�Q�                                    Bxap�D  "          A33@:=q�Dz�A33B�(�C��H@:=q����A�
Bi��C���                                    Bxap��  �          Az�@L(��2�\A��B��\C�ff@L(��x��A{Bl  C���                                    Bxap��  	�          A(�@Tz��,(�A
{B~z�C�|)@Tz��qG�A�Bj�C���                                    Bxaq6  "          A�@4z��6ffA��B�8RC�Q�@4z��}p�A�\Bo�HC��R                                    Bxaq�  T          A33@*�H�,(�A=qB��
C�aH@*�H�s33A�Bu{C��f                                    Bxaq)�  
�          A33@.{���HA��B�\C��@.{�7
=A��B���C��f                                    Bxaq8(  "          A@dz�Q�AQ�B�u�C�� @dz��z�Ap�B�=qC��                                    BxaqF�  
�          A{��Q���?J=q@���C�33��Q��Q쾙����p�C�7
                                    BxaqUt  �          A{�����?�{A�RC�H������>L��?�  C�\)                                    Bxaqd  
�          Aff��=q�
�R?���A
ffC����=q�Q�>aG�?�
=C��                                    Bxaqr�  T          A���\)��R?O\)@���C�˅��\)�33�\�C�˅                                    Bxaq�f  
�          A�����>�ff@4z�C�h�����333��(�C�e                                    Bxaq�  "          A33�������?5@�G�C�q��������G��333C�                                      Bxaq��  "          A(��\(��ff?W
=@�\)C�P��\(���H��33�	��C�S3                                    Bxaq�X  �          A�
�I���  ?�33A(�Cy���I���	>�z�?��Cy�f                                    Bxaq��  "          A�׿���R?��A
�\C�C׿��Q�>aG�?���C�T{                                    Bxaqʤ  
(          AQ����
=>�@7
=C�*=����H�333���
C�&f                                    Bxaq�J  
�          A\)��z�?�{@�{C)�����#�
���C8R                                    Bxaq��  	�          Aff����33@5A�ffC�\����  ?�Q�A$��C�                                      Bxaq��  
�          A����
��
?�Q�@��C�����
��ü��
��C�                                    Bxar<  
�          A������G�>L��?���C�������Q쿂�\����C��\                                    Bxar�  �          A�ÿTz��  ���8Q�C��H�Tz��ff��{��HC�z�                                    Bxar"�  
�          A��!G���R��=q��HC~5��!G��
=q�.�R���C}�R                                    Bxar1.  �          A(���{����=q� ��C�J=��{���!G��v�\C�"�                                    Bxar?�  "          A33��(���>�G�@'
=C�ٚ��(��33�O\)��G�C���                                    BxarNz  T          Aff��ff�Q�@A`Q�C����ff�  ?���@љ�C��                                    Bxar]   T          A�\�\���@�
AD��C�W
�\��?Q�@��C�o\                                    Bxark�  �          Ap���G���@!�At��C�J=��G���?��@��\C�j=                                    Bxarzl  �          A�����?��HA	G�C������G�>L��?�
=C�f                                    Bxar�  T          A�R���R��?���A*=qC����R�=q?��@Mp�C�*=                                    Bxar��  T          A=q��
=��\��
�^{C����
=�Q��`  ��ffC��=                                    Bxar�^  �          Ap���G��Q��ff�(z�C��׿�G��33�B�\��  C���                                    Bxar�  �          A33��{�녿�
=�7�C�!H��{����J=q���
C��{                                    Bxarê  �          A=q��{�녿�\�)G�C�޸��{����@  ��
=C���                                    Bxar�P  T          A33��=q���������C�C׿�=q���#�
�t��C�"�                                    Bxar��  �          A{�޸R�(����R�?33C����޸R��\�Mp���C�xR                                    Bxar�  �          A(��޸R�����m��C��f�޸R�	G��fff����C�Ff                                    Bxar�B  �          A(�������\���:�HC���������Ϳ����G�C�޸                                    Bxas�  
�          A�H���H���=�G�?#�
C�Q쿚�H�zΐ33�߮C�J=                                    Bxas�  "          A�H��{��׾�����C��3��{�ff���� ��C�                                    Bxas*4  �          A(���(���H���H�p�C����(��ff�+����C���                                    Bxas8�  T          A�R��33�����R�uG�C����33�
=�i����
=C���                                    BxasG�  �          Aff��z��Q��+�����C����z��G��s�
���C�XR                                    BxasV&  �          A����H�������c�C�N���H��R�[���
=C�R                                    Bxasd�  T          A{�(���\�����?�C���(�����HQ���=qC�                                    Bxassr  �          A� ���������_�C���� ���\)�\(����\C�S3                                    Bxas�  T          Aff������(����  C��f����qG����HC�Q�                                    Bxas��  �          A�R����  �a�����C��=�����{���
���
C�
                                    Bxas�d  "          Az�����
=q����p��CO\������c�
���HC~��                                    Bxas�
  �          A�����(���(��Dz�C
����\�I�����HC~�                                     Bxas��  "          A\)�p��	���(��v=qC�{�p���H�fff��  C~�H                                    Bxas�V  
Z          Ap��=q�\)��\)��(�C~�3�=q����g�C~�\                                    Bxas��  T          Ap��{��R��z���p�C~� �{�
�H�Q��lQ�C~
                                    Bxas�  T          A�
�����Ϳ�G��+33C�*=������?\)��
=C�{                                    Bxas�H  �          A���R�G������HC�)��R����(Q���
=C�\                                    Bxat�  
�          A33�  �zῼ(��\)C�  �  ��
�,����\)C��                                    Bxat�  T          A=q��R����   �EG�C~B���R�
=�L�����
C}��                                    Bxat#:  "          A
=����ff�
=q�S\)C����Q��W����C&f                                    Bxat1�  T          A��?\)��ÿ@  ��\)C{�?\)��   �A�C{c�                                    Bxat@�  �          A{�E�����R�l(�C{=q�E��R����4Q�Cz�f                                    BxatO,  �          A��,���������\)C}@ �,���33�%�~=qC|�                                     Bxat]�  
Z          Ap��\)�=q�����/\)C���\)����E���{C�                                    Bxatlx  �          A�����{��ff�-C�y����z��C�
���C�:�                                    Bxat{  �          A��������
=q�P��C�l����
�R�Z=q��\)C�#�                                    Bxat��  
�          A\)�����H�333���HC~�R�����
�����B�\C~��                                    Bxat�j  T          A�R�*�H�33?���@�=qC}h��*�H�zὣ�
��C}��                                    Bxat�  T          A��"�\��H?�p�@�C~!H�"�\�Q�L�;�\)C~E                                    Bxat��  �          A�R�{�(�?�Q�@�\)C~���{�p����Ϳ!G�C~Ǯ                                    Bxat�\  �          A����H�33?�Q�@��HC~�)��H�zὸQ���C~�q                                    Bxat�  �          A=q�,(���R?��@޸RC}0��,(���
�\)�aG�C}Q�                                    Bxat�  �          AG��   ��R?��@�ffC~T{�   ���L�Ϳ���C~p�                                    Bxat�N  T          A���33�33?Y��@�\)C�)�33����G��/\)C�f                                    Bxat��  �          A�����
>�
=@%�Cff��33�^�R����CY�                                    Bxau�  T          A���
�H�  ?8Q�@�ffC�9��
�H�(��z��eC�:�                                    Bxau@  �          A�� ���=q?��R@�z�C~33� �����u��{C~Y�                                    Bxau*�  
�          AQ�����R?��\@ə�C�H������\)��  C��                                    Bxau9�  T          A�
�<����?�A(��Cz�
�<���
{>�(�@,(�C{(�                                    BxauH2  �          A�u����
@p  A�=qCr�\�u����H@(��A���Cs�R                                    BxauV�  T          A���j=q��{@HQ�A��Ct�{�j=q��?�p�AICu��                                    Bxaue~  T          A���Z�H� ��@�RA}G�Cw�Z�H�G�?��
A��Cw�3                                    Bxaut$  
�          A=q�QG��
=@�Apz�Cx@ �QG��33?�33@�RCxٚ                                    Bxau��  �          A�\�^�R�=q@33Ah��Cv�)�^�R�=q?�=q@���CwxR                                    Bxau�p  
�          A�R�N�R� ��@:�HA��HCx#��N�R�ff?��HA,  Cx�3                                    Bxau�  "          A���E����@<(�A��Cy���E��
{?�Q�A'
=CzaH                                    Bxau��  
�          A���L�����@5�A��Cx�)�L���	?�=qA  Cy��                                    Bxau�b  �          A(��P  �=q@�AY�Cx�{�P  �	�?h��@�p�CyW
                                    Bxau�  "          A�H�<(��(�@0��A�=qCz��<(��G�?�(�A\)C{��                                    Bxauڮ  "          A�R�?\)�33@9��A�\)Cz�=�?\)�z�?�{Ap�C{:�                                    Bxau�T  
�          A���B�\�{@s33A���Cy}q�B�\�	@"�\Ay�Cz�=                                    Bxau��  
�          A�R�Z=q�G�?�A7�Cx޸�Z=q�Q�?z�@]p�CyE                                    Bxav�  
�          A(��XQ��z�?�\)A ��Cyu��XQ��=q    �#�
Cy��                                    BxavF  
�          A��`  ��?�z�A3�Cx��`  ��\?
=q@J�HCy�                                    Bxav#�  
�          A{�K��{@   A;\)Cz���K���?(�@a�C{.                                    Bxav2�  �          A��N�R�?fff@�=qCzxR�N�R�=q���7
=Cz��                                    BxavA8  �          A�R�8Q��(�>�(�@%C|B��8Q����p����{C|0�                                    BxavO�  
�          AG����33@[�A��C�Z�����@�\A?�C���                                    Bxav^�  �          Aff�(��Q�@\��A�
=C�/\�(��
=@�\A=�C�y�                                    Bxavm*  !          A
=��ff@C�
A�p�C����(�?У�A�RC�R                                    Bxav{�  
Z          A�H�	�����@!�Al(�C����	����?���@�{C���                                    Bxav�v  �          AG���
��
@��Af�RC����
�  ?�  @��C�+�                                    Bxav�  �          A�R��z��G�?�  A"�RC�^���z���>��?�G�C�u�                                    Bxav��  T          Ap����
�ff?��@�33C�:ΰ�
��
�#�
�n{C�C�                                    Bxav�h  �          A�(����?   @9��C�{�(���zῂ�\���RC��                                    Bxav�  �          A�
�Tz��
=>��R?�C��ͿTz��녿������C���                                    BxavӴ  �          A��>�\)�{��{��
C��q>�\)�(��B�\��G�C��f                                    Bxav�Z  �          Aff�u�\)���H�$Q�C�Ф�u���J=q��C��                                    Bxav�   �          A�    �녿�\)��=qC���    �G��$z��|��C���                                    Bxav��  �          A��k��{�k���{C�N�k����=q�g
=C�H�                                    BxawL  �          A\)��33���?��Az�C��R��33��R=�\)>���C��                                    Bxaw�  �          A�H�:=q�  @�{AᙚCz�{�:=q�=q@UA��C{޸                                    Bxaw+�  �          A��,�����@��\A�Q�C|� �,����@;�A���C}�=                                    Bxaw:>  �          A�
��(����@6ffA�
=C����(���H?��@�\)C�>�                                    BxawH�  �          A �Ϳ�(����@�\AS\)C�.��(����?@  @���C�Q�                                    BxawW�  �          A�\�(��=q@7�A���C\�(���?���A z�C�{                                    Bxawf0  T          A�G��{@�z�Aƣ�Cy���G���H@0  A���Cz�R                                    Bxawt�  �          A��G��
�R@s33A�G�Cz5��G���R@�A[�C{33                                    Bxaw�|  �          A Q��P  �\)@�  A��Cxff�P  ��
@(��A|z�Cy�)                                    Bxaw�"  �          A)��G��$�ý#�
�L��C��=�G��"�H��Q���C��{                                    Bxaw��  �          A*�R�7
=�#�>\@�\C~\)�7
=�"�\���
��z�C~B�                                    Bxaw�n  �          A+\)���'��=p��~{C������#����Q��C�ff                                    Bxaw�  �          A+�
����(�׿333�p��C�녿���$z�����N�HC���                                    Bxaw̺  �          A,  ���(�;��
��Q�C������%���-�C�xR                                    Bxaw�`  �          A,z��G��(z�.{�fffC���G��%���� ��C��3                                    Bxaw�  �          A-��ff�)p��#�
�X��C�ff�ff�%���
=�IC�E                                    Bxaw��  �          A+33��
=�(  ������C�uÿ�
=�"�R�3�
�s�C�Q�                                    BxaxR  �          A)녿�R�&�H��p���RC�K���R�   �Z�H����C�9�                                    Bxax�  �          A+33�����(�Ϳ�
=�Q�C�
�����"{�X�����C�                                    Bxax$�  �          A*�H�
=�)G������
=C�c׿
=�#\)�Dz����C�U�                                    Bxax3D  �          A+
=���)p������C����#��AG�����C��
                                    BxaxA�  �          A*ff�=p��)p��Y����=qC����=p��$���%�a��C��                                    BxaxP�  �          A*�R�fff�)p��k���C���fff�$���*�H�g�C��                                     Bxax_6  �          A,(��^�R�*�R��\)��  C���^�R�%G��8���yG�C���                                    Bxaxm�  �          A-���G��+
=��ff���C�P���G��%��Dz���  C�8R                                    Bxax|�  �          A*=q�p���&ff���&�HC�g��p����H�h����ffC�H�                                    Bxax�(  �          A(Q�#�
�"�\�%�d  C�33�#�
�G���Q���\)C�
                                    Bxax��  �          A*{�u�"�H�B�\���HC�W
�u�Q����R�ң�C�J=                                    Bxax�t  �          A(  ��Q�� ���B�\���RC��q��Q��{��{�ԣ�C��R                                    Bxax�  �          A)��ff�"ff�333�uC����ff�z�����ȣ�C�ٚ                                    Bxax��  �          A*ff�#�
�!��W����
C�1�#�
�����G���{C�                                    Bxax�f  T          A+33�\)�"�H�N�R����C�lͿ\)����p���33C�O\                                    Bxax�  �          A*�R�+��"�R�I����\)C��+�����33��  C��R                                    Bxax�  T          A+33����$���4z��t��C�޸�����R������\)C�˅                                    Bxay X  �          A*{��G���R��������C�����G������(��{C���                                    Bxay�  �          A*{�fff�$  �)���g\)C��H�fff�=q��(����HC�XR                                    Bxay�  �          A)�����\�!��;����\C�!H���\�\)��z����C��                                    Bxay,J  �          A'33��G����Z�H����C�Ϳ�G��G���=q��(�C�Ф                                    Bxay:�  �          A'��\���G����RC�� �\��H��G����C�L�                                    BxayI�  T          A'������33�r�\����C�y������=q����p�C�(�                                    BxayX<  �          A'
=���
��\�z�H��z�C�����
�G������ �\C��{                                    Bxayf�  �          A(  >�(����������\C�Ff>�(��
�R������RC�h�                                    Bxayu�  �          A)p���G��  ��33��ffC�����G��{��  ���C���                                    Bxay�.  �          A'��B�\���vff����C�s3�B�\�=q��Q���{C�ff                                    Bxay��  �          A)��Q��G��}p�����C�� ��Q�����z�� ��C���                                    Bxay�z  �          A((����
��\�XQ�����C�����
�ff���H���C��q                                    Bxay�   T          A#�?Tz��{���\���\C���?Tz���33�Ϯ�%�C�7
                                    Bxay��  �          A$��?��R� z���G���C�e?��R������(��?�\C�0�                                    Bxay�l  �          A�?�z���(����
�Q�C��\?�z���ff��{�8��C��                                    Bxay�  �          A(�?��
�����
=�6=qC�}q?��
����� ���^\)C�{                                    Bxay�  �          A
=?���������B�RC�n?�������Q��j�C�}q                                    Bxay�^  �          Aff?�
=����ڏ\�4=qC��?�
=��=q��p��\C��=                                    Bxaz  �          A�?�Q���\)��ff�1�
C��?�Q���������Z�
C�<)                                    Bxaz�  �          A�R>�����R����� \)C�)>�����=q��
=�*=qC�Ff                                    Bxaz%P  �          A\)�#�
��������ԏ\C�޸�#�
���
��Q��Q�C�ٚ                                    Bxaz3�  �          A��W
=�ff��\�>ffC��׿W
=��qG����C�^�                                    BxazB�  �          A=q?�z���p���p���
=C�t{?�z���{���R�)=qC�                                      BxazQB  �          A�@'���p������?�C�ff@'��������\�e
=C���                                    Bxaz_�  T          A��@�����
�ƸR�#33C��H@��������
�Kp�C�k�                                    Bxazn�  �          A%���0  �33?333@y��C~���0  ��\��33��z�C~}q                                    Bxaz}4  �          A%p��'��\)�u��{CB��'��z��z��*�HC~�q                                    Bxaz��  �          A&{���"=q�L������C�U�������*�H�n�RC�'�                                    Bxaz��  �          A'����H�#�����\C��쿺�H���b�\��z�C���                                    Bxaz�&  �          A'\)����"{�p��C\)C�^���������=q��G�C�&f                                    Bxaz��  �          A'���z��"�R�G��2{C�
=��z���x�����RC��\                                    Bxaz�r  �          A'\)��
=�!��(��W�
C��ÿ�
=��H��G����C���                                    Bxaz�  �          A(�ÿ���� ���@  ��=qC��׿��������33����C���                                    Bxaz�  �          A)���u�p���  ����C��ýu�����\)�(�C�Ф                                    Bxaz�d  �          A*�\=���\��Q���(�C�]q=��	���Q���HC�h�                                    Bxa{
  �          A((�=���\��ff��33C�aH=�����˅��\C�n                                    Bxa{�  �          A&�H�W
=�G���p����RC�]q�W
=���(��ffC�K�                                    Bxa{V  �          A(z��H��� Q�>�z�?ǮC|�{�H���=q�����C|^�                                    Bxa{,�  �          A)��`  ��R?z�H@��\Cz�\�`  ��R�p�����
Cz��                                    Bxa{;�  �          A((��Vff�{?��R@�  C{B��Vff��H�0���q�C{Y�                                    Bxa{JH  �          A)�N{�!G�?=p�@~�RC|B��N{� z῝p���z�C|.                                    Bxa{X�  �          A*ff����$�þu����C�y�����!��\)�C\)C�S3                                    Bxa{g�  �          A+\)���%���33��C��������g�����C���                                    Bxa{v:  �          A+��$z��%�fff���HC��$z����:=q�{�
C�=                                    Bxa{��  �          A,  �Q��(  >�(�@z�C�Ff�Q��&=q�У���
C�5�                                    Bxa{��  �          A-���\)�*{��ff��C�  ��\)�%G��!G��X(�C��)                                    Bxa{�,  �          A-p����+�?Q�@��
C�� ���*�H��ff���C��q                                    Bxa{��  �          A-녿����,Q�>Ǯ@�
C��H�����*{�޸R��
C�u�                                    Bxa{�x  �          A.�R��{�,�׾��ÿ�(�C���{�((��p��P  C��f                                    Bxa{�  �          A.{��)p��G����C��f��#��7��tz�C��\                                    Bxa{��  T          A.ff�
�H�*{�u���C�:��
�H�#��C�
��z�C��                                    Bxa{�j  �          A.�\���*=q�����\)C�^����#33�O\)��(�C�"�                                    Bxa{�  �          A/33���(z�ٙ��\)C��������qG����C�P�                                    Bxa|�  �          A/�
=#�
�%��j�H���\C�q=#�
�{��p���G�C�!H                                    Bxa|\  �          A.{�\�ff�����C����\����H�p�C�!H                                    Bxa|&  �          A,�׿�p���
������HC��Ϳ�p���p�����*��C��                                    Bxa|4�  �          A+��Q��33������RC���Q����H��{�0�C}�{                                    Bxa|CN  �          A*�H���������\��  C�33�����p�����(�C���                                    Bxa|Q�  �          A,�׿������������C�j=�����
{�ə���
C��
                                    Bxa|`�  T          A,���
=q�
=���R���C��
�
=q�
{��33�	C�                                    Bxa|o@  �          A,���G������Q���33C�O\�G���H���
�z�C~�f                                    Bxa|}�  "          A+��   ����(�����C���   ����  �C�\)                                    Bxa|��  
�          A,�׿��\)�}p���z�C�������R���z�C�G�                                    Bxa|�2  �          A,Q�:�H�#\)�Tz����C��{�:�H��������RC�                                    Bxa|��  �          A,Q�>�33�$Q��HQ�����C��
>�33�ff��ff��G�C�                                    Bxa|�~  �          A+��
=�$Q��E���\)C�Y��
=�ff������G�C�33                                    Bxa|�$  �          A-���H�&�R�@������C������H����������C��=                                    Bxa|��  �          A.=q�Ǯ�(Q��1��lQ�C���Ǯ�\)��p��֣�C���                                    Bxa|�p  �          A.�H��G��%p��]p���{C��=��G��{������HC��=                                    Bxa|�  �          A.{���
�'��6ff�s\)C������
�{��\)���HC���                                    Bxa}�  �          A.=q?xQ��*ff�G��*�HC��R?xQ���
��
=��z�C���                                    Bxa}b  �          A.�R?k��*{�z��D  C�|)?k��ff��Q���\)C��=                                    Bxa}  �          A/33?���(z��.�R�g\)C���?���\)������33C�\                                    Bxa}-�  �          A-�    �"�\�w����RC���    �����{��C���                                    Bxa}<T  �          A-��>���33��������C�l�>�������=q���C�z�                                    Bxa}J�  �          A-��0�������H���C�׿0������33�
=C��                                     Bxa}Y�  �          A,�ÿQ��!��r�\��z�C��ͿQ��Q�������C�g�                                    Bxa}hF  �          A-����� �����H���C��f����H����
G�C���                                    Bxa}v�  �          A-p�=u�������ɅC�(�=u�	G����
���C�/\                                    Bxa}��  �          A.{�#�
������Q�C��H�#�
�
{��(��(�C��q                                    Bxa}�8  �          A.=q>�(��!G����H��ffC�5�>�(��33���
Q�C�]q                                    Bxa}��  �          A-�>��R�{��  ���C���>��R�
�\��G��=qC��                                    Bxa}��  �          A-��>�����������\)C��q>��������z���C��q                                    Bxa}�*  �          A-�������G���ff��=qC�������Q���p���RC���                                    Bxa}��  �          A-G��   ���(���RC�z�   ��ff�陚�*
=C�7
                                    Bxa}�v  �          A,�ÿ8Q��p���(���
=C����8Q���p�����*G�C�c�                                    Bxa}�  �          A,�׿�33��\��ff���HC�箿�33��\)���H�+�\C��q                                    Bxa}��  T          A.{��\�\)��ff�أ�C�����\�{��p��33C�f                                    Bxa~	h  �          A/33��\��������C����\�	p���\)�  C�P�                                    Bxa~  �          A/33�)���
=���
��G�C~L��)������33��C{�f                                    Bxa~&�  �          A/��,���  �����θRC~#��,���
=�����C{�                                    Bxa~5Z  �          A/���\�ff�����ffC�}q��\����\)�
�CL�                                    Bxa~D   �          A.�R�*=q�(����
��33C~���*=q�	G���ff�
z�C|                                    Bxa~R�  �          A/
=��R�   �h����C�
=��R��R����� G�C~z�                                    Bxa~aL  �          A/\)�ff�!G��qG����RC���ff����ff�ffC�j=                                    Bxa~o�  �          A/
=��Q��!���n�R����C��\��Q���
����C��                                    Bxa~~�  T          A.�H����#33�_\)���HC������=q��
=��=qC�y�                                    Bxa~�>  �          A.�H���"�R�\(�����C�Z�������p����C���                                    Bxa~��  �          A.�R���R�$z��U���RC�� ���R��
��33��33C�k�                                    Bxa~��  �          A.�H�Q��"ff�|(���
=C���Q�������	Q�C�aH                                    Bxa~�0  �          A-��?�ff�����33���HC�ٚ?�ff�����
=C�|)                                    Bxa~��  �          A-G�?��\�������C�H?��\�{�أ���
C�y�                                    Bxa~�|  �          A-G�?�������33��p�C�"�?����H����C���                                    Bxa~�"  �          A,��?�G����������C�|)?�G�����ff���C�1�                                    Bxa~��  �          A,��?���������噚C�g�?������
��{�'��C�4{                                    Bxan  �          A+�
?����\��{�ۮC��?���   ��  �#�C���                                    Bxa  �          A,  ?�����\��(���z�C�޸?����������
�-ffC��\                                    Bxa�  �          A,��?���z������ffC��?������Q��!��C��R                                    Bxa.`  �          A,��?������z��㙚C���?����H��{�&��C���                                    Bxa=  �          A*�H@�ff��=q���HC�j=@��ff��\�&\)C��3                                    BxaK�  �          A+�
?����(����H�˙�C��?�����\�ָR���C���                                    BxaZR  �          A&{�������u����
C�9������R��  ���C���                                    Bxah�  �          A#�@/\)�p���=q��z�C�]q@/\)�ڏ\�Å��HC��                                    Bxaw�  �          A"�H@��\������������C��H@��\��{�Ϯ���C���                                    Bxa�D  �          A#33@�p��=q��z�����C��@�p��ۅ�ƸR��
C��                                    Bxa��  �          A#�
@�\)�  �_\)��C��@�\)��p���� =qC���                                    Bxa��  T          A"�H@�����Dz���=qC��
@����\)��=q��33C�`                                     Bxa�6  �          A"�H@w��
=�?\)���RC��{@w���ff�����(�C���                                    Bxa��  �          A!��@q��33�#33�k
=C�p�@q�����z���C��=                                    Bxaς  �          A%p�@�=q����p���(�C��@�=q��R�S�
��{C���                                    Bxa�(  �          A$��@��
�������   C���@��
��p��l����ffC��                                     Bxa��  �          A ��@�z��Q����$��C���@�z����H�p  ��C��\                                    Bxa�t  �          A!p�@2�\����   �lQ�C���@2�\��R��(����
C�n                                    Bxa�
  �          A)����G��
=��p����CrY���G���p�����&{CmB�                                    Bxa��  T          A)��g
=�
�\��
=���Cw\)�g
=����{�#Q�CsJ=                                    Bxa�'f  �          A&�H���H�����R��p�Crp����H��
=���
�Q�Cn\                                    Bxa�6  �          A'\)�tz��G��}p�����Cv�)�tz���\��Q��Q�CsB�                                    Bxa�D�  �          A)��8Q��\)�������C|���8Q����H��G��Q�Cy�{                                    Bxa�SX  �          A)�G��  �^{��
=C{�{�G��p���
=���Cy��                                    Bxa�a�  �          A,z��N{�\)�W���  C{���N{�	������RCyff                                    Bxa�p�  �          A,z��G�����O\)���C|Q��G��
�H���\��p�Cz5�                                    Bxa�J  �          A*=q�\)�\)�\(����C�H�\)�����Q���RC}                                    Bxa���  �          A$���*=q��H�K����C~8R�*=q�G���{��G�C|B�                                    Bxa���  �          A$Q쿆ff�
=��\)��C�����ff��R�ڏ\�(�C�                                      Bxa��<  �          A\)��{�
=�0����=qCtE��{���7
=��p�Cs�                                    Bxa���  �          A33�a���+��z�HCxL��a��������\��Cu�                                    Bxa�Ȉ  �          A   �G��z��7
=��p�Cz��G�� (�������
Cx�3                                    Bxa��.  �          A��.{� ����z���G�C{T{�.{��33�����)�HCw��                                    Bxa���  �          A��
=���������Cuÿ�
=���
���H�Sp�C{\                                    Bxa��z  �          A!���H� �������p�C�(����H�ə���(��D�\C|�q                                    Bxa�   �          A#�
�����\��z���C��\�����p���Q��'p�C���                                    Bxa��  �          A"�\��(��p��Å�
=C�~���(���  ��\)�Oz�C��                                    Bxa� l  �          A"�\>.{��H��  ��C���>.{��33����L�
C��                                     Bxa�/  �          A"ff�
=����z��33C��
�
=��p�����I�HC�e                                    Bxa�=�  �          A!�����������R��
=C��
�����z���  �7�RC��
                                    Bxa�L^  �          A!��{�(�������HC��\��{��33��{��C�                                    Bxa�[  �          A"{��������
����C��ÿ����G�����!��C�޸                                    Bxa�i�  �          A$Q���H��K�����C��
���H�
=���\�C�J=                                    Bxa�xP  �          A#���  ���G
=����C�����  ����Q���C���                                    Bxa���  �          A$Q쿰���p��X����ffC��=�������G��	\)C�S3                                    Bxa���  �          A$(��������U����C�uÿ�����\����=qC��{                                    Bxa��B  �          A$Q쿨�����@  ��{C�&f�����	�����R� �\C��f                                    Bxa���  T          A%���Q��  �C�
���
C����Q��	p�������C�@                                     Bxa���  �          A&�H��33����Z=q����C��f��33�z���(��	Q�C�(�                                    Bxa��4  �          A)G��xQ�����u���  C�/\�xQ��
=�ə��G�C���                                    Bxa���  �          A)�W
=�������{C�b��W
=�G���z��!��C�C�                                    Bxa��  �          A)���p�����^�R��{C�����p��(���
=�	�C�AH                                    Bxa��&  �          A+\)�����!��<����C�������\)��������C�޸                                    Bxa�
�  �          A+�
��\� ���X����\)C�����\�(����R��HC�G�                                    Bxa�r  �          A,z�У���
�l������C�Z�У��
{��  ���C���                                    Bxa�(  �          A*�R?������=q���HC��\?���ff����<�HC�E                                    Bxa�6�  �          A'\)?����z���ff�
�C��3?�����33� z��H�
C���                                    Bxa�Ed  �          A(  ?�\)����Q��
=C��R?�\)�У��G��I��C��                                    Bxa�T
  �          A'�
?�(��	p����\�(�C��q?�(�����ff�E{C�7
                                    Bxa�b�  �          A(z�=�\)������H��Q�C�5�=�\)��Q����9ffC�C�                                    Bxa�qV  �          A(Q쿬��������C��\������G���z��;\)C��
                                    Bxa��  �          A(�׿��H�z���
=���\C��ÿ��H��{��p��:��C~@                                     Bxa���  �          A)G��$z��
{�������C}h��$z��أ���ff�;
=Cy5�                                    Bxa��H  �          A)���$z���������
=C}�\�$z������
=�3Q�Cz                                    Bxa���  �          A(z��Q��ff���
��(�Cff�Q�����޸R�%�RC|h�                                    Bxa���  �          A(  ��G�����������C����G�����G��(�C��{                                    Bxa��:  �          A(�׿�=q�!��,���n�\C�:Ὺ=q��R��z�����C��H                                    Bxa���  �          A%���=q�33��p���
C�<)��=q����33�R�C��                                    Bxa��  �          A&{����z������{C�e����ȣ��p��M33CW
                                    Bxa��,  �          A'��{���Q����C��{��p����
�D�RCz�                                    Bxa��  �          A(z�����H��33���C~E����ff��
=�D�Cy�\                                    Bxa�x  T          A(  �!��	���R���\C}���!���\)����;=qCyk�                                    Bxa�!  �          A)G��2�\�
�\�������C|��2�\��G����
�8=qCw��                                    Bxa�/�  �          A)G��Fff�	p���33����Cz+��Fff��\)���6(�CuL�                                    Bxa�>j  �          A*ff�HQ��
�H���\��=qCz.�HQ���=q��=q�5
=CuY�                                    Bxa�M  �          A*=q�I���
�\��G�����Cy���I���ٙ������4z�Cu�                                    Bxa�[�  "          A,���q��ff�����ffCu�q���ff����9
=Co��                                    Bxa�j\  �          A+33�N{�	���{���
Cyn�N{��p�����7�Ct@                                     Bxa�y  �          A'
=�33��H��{��RC��=�33������G��1�C~�                                    Bxa���  �          A&ff�У��p�������z�C�녿У���33��p��.z�C��                                     Bxa��N  �          A&ff�У������33�ř�C���У���p��ڏ\�$G�C�                                      Bxa���  �          A&{�����H�e����C��H�����
�ƸR�{C��                                    Bxa���  �          A#
=��������~�R��(�C���������  �Ϯ��C��R                                    Bxa��@  "          A$�Ϳ���ff��Q����
C�녿����R�޸R�*G�C��                                    Bxa���  �          A'�����p����\��{C~�
����߮��p��4z�C{
                                    Bxa�ߌ  �          A%녿�=q����=q��33C�O\��=q��������*�C��                                    Bxa��2  �          A$�Ϳ�\)�z�������G�C�LͿ�\)��ff��=q��RC�+�                                    Bxa���  �          A$  �z��G��j�H��{C�� �z���33��\)�  C@                                     Bxa�~  �          A%��.�R�  �s33���C}���.�R������H��RCz�\                                    Bxa�$  �          A%��<����H�l�����C|+��<����ff��\)�Q�Cy                                      Bxa�(�  �          A%�3�
���p����Q�C}��3�
��
=����  Cy�R                                    Bxa�7p  �          A$���>�R����p  ��
=C{���>�R���H��Q��
=Cx��                                    Bxa�F  �          A&=q�A��=q�r�\��  C{���A����
��=q���CxO\                                    Bxa�T�  �          A'�
�k���\��  ��=qCw�=�k���  ��{���Cs{                                    Bxa�cb  �          A"=q�G���H�;����RC�>��G��{��33�(�C~p�                                    Bxa�r  �          A z��=q�ff��(��5p�C�� ��=q�
{��  ��Q�C�H                                    Bxa���  �          A$���������C\)C���������  ��\C�3                                    Bxa��T  �          A%�0  ���Y�����C}�)�0  ����G����Cz�
                                    Bxa���  �          A&{�)��������H���
C}�f�)����\��33�%33Cz)                                    Bxa���  �          A%��3�
�z���=q��C{�q�3�
���
�����7G�Cv�3                                    Bxa��F  �          A$z��P  ����  ����Cx�=�P  ��\)��G��4�\CsaH                                    Bxa���  �          A'33�&ff��H��  ��Q�C}��&ff���H��
=�.�
Cy�)                                    Bxa�ؒ  �          A'�
�0���33������
C|^��0���׮���8{Cw��                                    Bxa��8  �          A&�R�  �{����ݮC��  ��Q�����2��C|0�                                    Bxa���  �          A%p���R�Q���G����C�9���R������   C}��                                    Bxa��  �          A$���=q�p��������C+��=q������\)�#p�C{�R                                    Bxa�*  �          A&=q�33������z�C�)�33���
��{�/p�C|
                                    Bxa�!�  �          A&�\�
=�
=�������HC:��
=�ᙚ��G��1��C{�                                    Bxa�0v  �          A&�H�(���  ��������C}�f�(������\�+�Cy�\                                    Bxa�?  
�          A'��   �p�������33C~���   �����33�*�HCz�q                                    Bxa�M�  �          A(������(�������  C�}q�����p�����+Q�C}�f                                    Bxa�\h  �          A(���(������33����C~�H�(��������2{C{�                                    Bxa�k  �          A(Q��'��Q������ׅC}���'�����\�0(�Cy                                    Bxa�y�  �          A*�\�2�\�{��  �ӅC}  �2�\��R���H�-��Cx�
                                    Bxa��Z  �          A*{�+�����=q�˙�C}Ǯ�+���33��ff�*�Cy�3                                    Bxa��   �          A,  �7
=�33�������C|�R�7
=��Q������-�Cxz�                                    Bxa���  �          A,(��N�R����ff��G�Cz33�N�R��  ��  �0�HCu0�                                    Bxa��L  �          A,���Y���{��z���
=Cy\�Y���ڏ\�����4G�Cs�\                                    Bxa���  �          A(Q�����33��ff��(�CoY����������G�� ��Ch�=                                    Bxa�ј  �          A)���H����33��  Co�\���H���
��ff�#�Ch�\                                    Bxa��>  �          A-���=q�
=������=qCq�{��=q����ff�-�Cj��                                    Bxa���  �          A)����\��=q��33����Cj�)���\���\��
=�,z�CbQ�                                    Bxa���  �          A.�R���������=q���Cl�������
��\)�4��Cc�                                    Bxa�0  �          A1��p�������z�� �Ci���p�������
=�7G�C^��                                    Bxa��  �          A-p���=q�  �������HCq���=q��z���z��3�
CiaH                                    Bxa�)|  �          A0�������p���  ��(�Cl�������  ���-G�Cd�\                                    Bxa�8"  �          A2�\��ff�\)�����
=Cpk���ff��ff���&��Ci�\                                    Bxa�F�  �          A0����(��
=������HCq=q��(��ȣ���=q�4\)Ci�\                                    Bxa�Un  �          A/
=��� z���Q���(�Co�q������� Q��<�RCf�                                    Bxa�d  �          Az���
�33��z��<Q�C��H��
���
����33CL�                                    Bxa�r�  �          A
=��G���ÿQ����C��\��G����k����C��f                                    Bxa��`  �          A  ������p����C�z����Q��L�����HC�`                                     Bxa��  �          A   �G�����3�
���HCz�R�G���p����\��
Cw�
                                    Bxa���  �          A!G��(Q���H��W
=C~ff�(Q���R��Q����
C|!H                                    Bxa��R  �          A\)�(���\)��p����C~h��(�������R��G�C|�3                                    Bxa���  �          Ap�?W
=��ff?�=qATQ�C�  ?W
=��(��E�����C�                                    Bxa�ʞ  �          A�@�  ?�z�@�BV�Aep�@�  ��z�@�
=BWC�f                                    Bxa��D  �          A  @�(�?s33A�B_��A\)@�(���p�@�{BZ33C��                                    Bxa���  T          A��@�Q�?�Ap�B^p�@�ff@�Q����@�=qBS��C�c�                                    Bxa���  �          AQ�@��H>��A�RBb��@�G�@��H�{@�(�BV�HC���                                    Bxa�6  T          A�
@���=���A�\Bd{?��@����%�@�  BS
=C��                                    Bxa��  �          A�H@�Q�L��A�BdG�C��@�Q��6ff@�BOG�C��H                                    Bxa�"�  �          Aff@�Q�W
=A�
Bj��C���@�Q��9��@��RBTffC��
                                    Bxa�1(  �          A�@�G�?�RA��Bj  @��@�G����A ��B^�
C�|)                                    Bxa�?�  �          Az�@���>�(�A��Bh\)@�Q�@�����@�\)BZ�C�Ф                                    Bxa�Nt  �          Ap�@��>�z�Az�Be(�@I��@���p�@��BV
=C���                                    Bxa�]  �          A
=@�Q�=��
A=qBd�\?Tz�@�Q��'�@��RBR�C��q                                    Bxa�k�  �          A�@�  >#�
A�RBe{?��H@�  �#�
@�Q�BT�C��)                                    Bxa�zf  T          A�H@�G��aG�ABcp�C��@�G��9��@�=qBM�
C��q                                    Bxa��  �          Aff@�=q�\A ��BaC��@�=q�B�\@�RBJQ�C�!H                                    Bxa���  T          A��@�{�+�A   BcG�C�U�@�{�S33@�=qBG�HC��                                    Bxa��X  T          Az�@�p��W
=@��RBb�RC�b�@�p��\��@�
=BE(�C��                                    Bxa���  �          A33@�����@�(�BZ�C��@���8Q�@�\BEG�C�c�                                    Bxa�ä  �          A�@�(�>8Q�@�BZff?�ff@�(��   @���BJ�
C��                                    Bxa��J  �          A�@�ff>#�
@�(�BX\)?��@�ff�   @�\)BH�
C�.                                    Bxa���  �          A
=@���=�Q�@���BU�?\(�@����"�\@�33BEffC�#�                                    Bxa��  �          Aff@�(�    @��\BY�C���@�(��)��@��
BGG�C�h�                                    Bxa��<  �          A�H@�ff?�R@�BOG�@��@�ff���R@��
BFp�C��                                    Bxa��  �          A�R@�  ?.{@�G�BM33@�
=@�  ��@�\BE33C�&f                                    Bxa��  �          A@�p�?\)@�G�BN��@�z�@�p���\@���BE=qC���                                    Bxa�*.  �          A@�Q�=���@�p�BTff?xQ�@�Q��!G�@�  BDG�C�0�                                    Bxa�8�  �          A��@�G���p�@�Ba  C��q@�G��C33@�\BH�
C�                                      Bxa�Gz  �          A�R@�{�Ǯ@��Ba�HC��
@�{�C33@�  BI(�C��f                                    Bxa�V   �          A�@�p��Y��A (�BmQ�C�ٚ@�p��b�\@�\)BK��C�o\                                    Bxa�d�  �          A��@�G���ABr�\C��{@�G��N�R@�ffBT�
C�AH                                    Bxa�sl  �          A�@�
=����@�Bg\)C���@�
=�E@��BM=qC��                                    Bxa��  �          AG�@��
��33@��RBi�C���@��
�C�
@�33BO�
C���                                    Bxa���  �          AG�@�(��\@�ffBiz�C��=@�(��E@�\BO  C�Ф                                    Bxa��^  �          A�
@�ff��{@�{BmG�C��@�ff�Dz�@��HBR\)C�p�                                    Bxa��  �          A�@��;B�\@�=qBg��C�� @����8��@��BO�
C�                                    Bxa���  �          A�H@�����  @�  BZ�
C���@����5�@�ffBD�\C���                                    Bxa��P  �          Aff@��=�\)@��BRG�?0��@���p�@��HBAffC��                                    Bxa���  �          A@�>�Q�@�BM(�@hQ�@��Q�@�G�B@C���                                    Bxa��  �          Az�@�z�?5@ٙ�BC��@�(�@�z�ٙ�@��
B<��C��\                                    Bxa��B  �          Ap�@�{?�(�@�(�B8�
A^=q@�{�Y��@ϮB=
=C���                                    Bxa��  �          AQ�@���?�  @�G�B733A33@��ÿ�G�@ǮB5z�C�)                                    Bxa��  �          A�@�
=?�z�@ÅB2z�Ax��@�
=�
=@��B9��C�0�                                    Bxa�#4  �          A
=q@�33?��@��\B*{A��R@�33��p�@��HB3�C�B�                                    Bxa�1�  �          A
�R@�ff?�ff@\B3�
AE�@�ff�k�@���B6p�C���                                    Bxa�@�  �          A
=@���?У�@�
=BC33A���@��ͿW
=@ۅBH�C�Ǯ                                    Bxa�O&  �          A	G�@�
=?��@�B�
A{@�
=�B�\@�  B(�C�Ǯ                                    Bxa�]�  �          A	�@�33?s33@��
B$��A(�@�33����@�33B#�
C�R                                    Bxa�lr  �          A��@�=q?�G�@�=qBA\)@�=q�p��@��\B(�C��=                                    Bxa�{  �          A�
@��R��G�@���B5=qC�u�@��R�\)@��HB%33C���                                    Bxa���  T          Az�@�\)>���@��B+�@2�\@�\)���
@�G�B"�C�f                                    Bxa��d  
(          A��@�p���{@�{B2�\C��f@�p��O\)@�z�B�HC�&f                                    Bxa��
  �          AG�@�\)�xQ�@�{B1�C��{@�\)�G
=@�{B��C���                                    Bxa���  �          A�@�=q����@�
=B4�
C��)@�=q�Q�@��B\)C��3                                    Bxa��V  �          AQ�@����  @�=qB.��C�#�@���dz�@�(�B
=C��\                                    Bxa���  �          A��@��
��  @ƸRB433C�` @��
�J=q@�{BG�C�\)                                    Bxa��  �          A�@vff��G�@�G�B"�RC�c�@vff���
@Mp�A��RC��{                                    Bxa��H  �          A
=@&ff���H@�z�B��C���@&ff��33@��Au�C��                                     Bxa���  �          A
=@��\�z�H@�\)B6�RC��f@��\���@�p�A��C�T{                                    Bxa��  �          Aff@�����@�ffB>�RC�j=@����\)@�p�B��C�c�                                    Bxa�:  �          A
=@���1�@�33B3  C��@������@�B�
C�k�                                    Bxa�*�  �          A
�H@��H�Q�@���B'p�C�^�@��H���\@��
B �HC��                                    Bxa�9�  �          A
{@�
=�޸R@�\)B&Q�C�,�@�
=�l��@��RB33C��                                     Bxa�H,  T          A	p�@�=q���@�p�B%\)C�:�@�=q�P��@�=qBp�C�XR                                    Bxa�V�  
�          A
=@�����@�\)B%�C�(�@��G
=@�{B
��C�q                                    Bxa�ex  �          A	G�@�=q�xQ�@�\)B'��C��f@�=q�?\)@��B{C�W
                                    Bxa�t  �          Az�@ƸR�E�@�G�B+  C�y�@ƸR�5�@��B=qC��                                    Bxa���  �          A(�@�z�Tz�@��B#33C�N@�z��333@��
BC�'�                                    Bxa��j  �          A(�@ə��!G�@�B'�C�  @ə��*�H@��B�C���                                    Bxa��  �          AQ�@�z�u@�=qB$33C��q@�z��@��B�C���                                    Bxa���  �          A
=@�  ?(�@��HB=q@��H@�  ��33@�{B
=C��                                    Bxa��\  �          A�@��>��@�33B�@g
=@�녿�=q@�(�B  C�5�                                    Bxa��  �          A
=@�
=>�33@��\BQ�@<��@�
=��ff@�33B�\C��                                     Bxa�ڨ  �          A\)@�
=��G�@�B�C��@�
=��H@�(�B  C��                                    Bxa��N  �          A�@�G���ff@�z�B�
C��@�G��=q@�33BQ�C��                                    Bxa���  �          A�
@�(���
=@���B�HC�5�@�(��@�  B(�C�G�                                    Bxa��  �          A�H@љ���G�@�G�B\)C��@љ���@�  B	
=C�                                    Bxa�@  T          A\)@�{=�\)@���B6
=?&ff@�{��@�z�B&�
C��=                                    Bxa�#�  �          A
{@��=�Q�@���B5�R?Tz�@���p�@��B&C���                                    Bxa�2�  �          A�H@�
==��
@�ffB=G�?Q�@�
=��R@���B-33C�U�                                    Bxa�A2  �          A�H@�
=?E�@˅BD=q@�\)@�
=��z�@�{B=(�C���                                    Bxa�O�  �          A
=@��>�=q@��B/��@%@�녿�
=@���B$�C�&f                                    Bxa�^~  �          A�\@�(�>\)@���B/=q?�\)@�(���p�@�G�B!�RC���                                    Bxa�m$  �          A@��H>��@��B/z�?�G�@��H���H@�Q�B"{C��
                                    Bxa�{�  �          A�@�ff=u@���B5�?(�@�ff�
=@�(�B&�C���                                    Bxa��p  �          @��H@�G���R@�(�B
=C�3@�G��=q@�G�B�\C�%                                    Bxa��  �          @��
@�녿5@�
=B�
C��@���!�@��HB�HC���                                    Bxa���  �          @�@�����@�
=B�HC�
@����@���BQ�C�%                                    Bxa��b  �          @�\)@�����R@�z�BC�� @�����@��B
��C�.                                    Bxa��  �          @�
=@����8Q�@��RB(�C��@�������@�G�B�C���                                    Bxa�Ӯ  
�          @�z�@��
�#�
@���B!�C���@��
����@��B�RC��f                                    Bxa��T  �          @�p�@���z�@��B�C���@���
@���B{C��                                    Bxa���  �          @�Q�@�33��G�@��HB�C�)@�33�+�@w�A�  C��{                                    Bxa���  �          @�{@��� ��@�Q�B�C��
@���^�R@K�AîC�{                                    Bxa�F  �          @��R@�{����@�p�B��C�3@�{�+�@k�A�
=C�L�                                    Bxa��  �          @�(�@�녿&ff@��B  C��=@���p�@�  BC���                                    Bxa�+�  �          @��@�=q=��
@�G�B(z�?\(�@�=q����@�{Bp�C��{                                    Bxa�:8  �          @�\)@�(�?G�@�33B7�A	��@�(���=q@��B2�
C��R                                    Bxa�H�  �          @�(�@�(�>�@���B6z�@�z�@�(��˅@�G�B,�RC�c�                                    Bxa�W�  �          @陚@�(�?�G�@�  B:
=A:ff@�(���=q@�\)B9Q�C���                                    Bxa�f*  �          @�{@���?
=q@��BK�@�\)@��ÿ��@��BAffC�)                                    Bxa�t�  �          @��
@��H��@�
=BT�C���@��H�!G�@��
B5�\C�(�                                    Bxa��v  �          @�33@��?@  @�\)BHffA�H@�녿���@��
BB�\C�}q                                    Bxa��  �          @޸R@���?��H@��BD=qA�  @����+�@���BL\)C��\                                    Bxa���  �          @�33@QG�@%�@��BV�B�
@QG�>B�\@ƸRBx�@W
=                                    Bxa��h  �          @�(�@8��@Mp�@�{BS{B@(�@8��?B�\@ϮB��=Al��                                    Bxa��  �          @�
=@H��@_\)@�(�B>{B@=q@H��?�p�@\Bs��A�                                    Bxa�̴  �          @��@6ff@�@�  Bq��B@6ff���@��HB��C�P�                                    Bxa��Z  �          @��@{@#�
@ӅBv�HB7��@{�W
=@�\B�ffC��                                    Bxa��   �          @��@&ff@�ff@���B'��By=q@&ff@&ff@��Br�HB4G�                                    Bxa���  �          @�G�@(�@���@��
B��B�Ǯ@(�@R�\@��Biz�Baff                                    Bxa�L  �          @��@(��@��R@�z�B��B�G�@(��@S�
@�p�B^{BM��                                    Bxa��  �          @�z�?�{@�=q@EAυB��H?�{@�(�@�
=B>\)B��                                    Bxa�$�  �          @�Q�@@!�@��HBn=qBI��@>��@�z�B�B�@\)                                    Bxa�3>  �          @�\@��@e�@�=qBPz�B^p�@��?�{@У�B�8RA�33                                    Bxa�A�  �          @�@C33@fff@�(�BC(�BF��@C33?���@˅B{
=A���                                    Bxa�P�  �          @��@o\)@1�@�(�BIz�B�@o\)>�33@ȣ�BlQ�@��
                                    Bxa�_0  �          @���@e@���@�p�B'=qBNp�@e@��@�G�Be��A��                                    Bxa�m�  �          A z�@fff@�p�@�\)B"��BWQ�@fff@p�@�  Bd�B	��                                    Bxa�||  �          @�?G�@�33���R�z�B�Q�?G�@�@-p�A�{B��                                    Bxa��"  �          @���>�Q�@�33��ff�Y��B���>�Q�@��@%�A�z�B�L�                                    Bxa���  �          @�G�>aG�@�{�u��33B�Ǯ>aG�@�{@
=qA�33B���                                    Bxa��n  �          A�?�
=@�\)��Q����B��?�
=@���@�
AiG�B�8R                                    Bxa��  �          A Q�?�
=@��ÿ�����B���?�
=@�=q@Aq�B�                                      Bxa�ź  �          @��R?ٙ�@�
=�u��p�B��?ٙ�@�R@�A~=qB�L�                                    Bxa��`  �          @�p�?˅@�
=�@  ��p�B���?˅@�(�@Q�A��
B���                                    Bxa��  �          @�33?�\)@�p������<(�B�#�?�\)@�ff@+�A���B���                                    Bxa��  �          @��H?���@�{�
=��=qB�(�?���@�G�@!G�A�=qB�{                                    Bxa� R  T          @�Q�?�p�@��
>��?��B�\?�p�@�p�@K�A�G�B��                                    Bxa��  T          @�33?�  @�
=>��@Dz�B�\?�  @�{@W
=A��B��f                                    Bxa��  �          @�ff?���@���?333@��
B�� ?���@�z�@c33A�G�B�z�                                    Bxa�,D  �          @�ff?�{@�\)?8Q�@�G�B��)?�{@��H@c33A��B�u�                                    Bxa�:�  �          @�?�
=@�33>���@   B���?�
=@�33@O\)A�
=B��{                                    Bxa�I�  �          @�(�?���@�G��B�\���HB�
=?���@߮@6ffA�33B��3                                    Bxa�X6  �          @��H?���@����
���B��q?���@�z�@;�A��B�.                                    Bxa�f�  �          @��
?���@��þB�\���HB�Q�?���@�\)@6ffA�B�                                      Bxa�u�  �          @�?aG�@�녽��
��RB��3?aG�@�ff@>{A�p�B��\                                    Bxa��(  �          @�\?��@������z�B��?��@�=q@!G�A�
=B�33                                    Bxa���  �          @��?��@��Ϳ333���HB��?��@��@ffA���B��f                                    Bxa��t  �          @���?�z�@�
=?333@���B�.?�z�@�33@\��A���B��=                                    Bxa��  �          @׮?�(�@Ӆ>�{@8��B�\?�(�@��@>�RA�Q�B���                                    Bxa���  �          @׮?��@��}p��z�B�k�?��@���?�Q�AiB�Ǯ                                    Bxa��f  �          @�Q�?���@�(����H�$��B��?���@�=q?�p�AJ�HB��
                                    Bxa��  �          @ۅ?�ff@�(���p��j�\B��=?�ff@�G�?�  A�\B�=q                                    Bxa��  �          @أ�?�33@Ǯ�33���HB�L�?�33@���?(��@�33B��{                                    Bxa��X  �          @�
=?�@��
�����
=B��?�@��>���@"�\B���                                    Bxa��  �          @���?�@�  �:�H��{B�=q?�@Ǯ?�p�A�ffB�{                                    Bxa��  
�          @�@ff@�{?���A#33B�z�@ff@�Q�@��B�B��{                                    Bxa�%J  �          @�\?�z�@�Q�?��\@�Q�B��{?�z�@�
=@y��A�p�B��H                                    Bxa�3�  �          @�\?���@陚?^�R@��B�{?���@��@qG�A��HB��                                    Bxa�B�  T          @�(�?޸R@�(�?Tz�@ǮB�u�?޸R@�z�@qG�A��B��\                                    Bxa�Q<  T          @�33?У�@��
?^�R@љ�B�?У�@��
@s�
A�z�B�=q                                    Bxa�_�  �          @�
=?�Q�@�\)?Tz�@���B��=?�Q�@Ϯ@tz�A�\)B���                                    Bxa�n�  �          @��R?��@�{?Tz�@�p�B���?��@�ff@s�
A���B���                                    Bxa�}.  �          @�
=?�
=@�?J=q@��B��?�
=@�  @s33A��
B�                                      Bxa���  �          @�{?�=q@��?xQ�@��B�G�?�=q@�33@{�A��
B��f                                    Bxa��z  T          @���?�@�R?��A�\B��R?�@�Q�@�Q�B��B�                                    Bxa��   �          @�\)@ ��@�\?��A&ffB�aH@ ��@�33@�G�B�
B���                                    Bxa���  �          @��@
=@�33?�  A733B��@
=@�33@�G�B
z�B�aH                                    Bxa��l  �          @�Q�@�
@��?��A=��B�u�@�
@�G�@�=qBQ�B��
                                    Bxa��  �          @�R@�\@���?�ffA?�B���@�\@�Q�@�=qB��B�                                      Bxa��  �          @���@:�H@��?�Af�\B���@:�H@��H@���B�HBpp�                                    Bxa��^  �          @�{@333@��?У�AQ�B��
@333@�p�@�(�Bz�Bv33                                    Bxa�  �          @���@5�@У�?��ADQ�B��@5�@�G�@��B	�\Bw�\                                    Bxa��  �          @���@,��@��?��AP(�B�L�@,��@�G�@��RB\)B{��                                    Bxa�P  T          @��@�R@�\)?��
Ae�B��@�R@��@��B�
B���                                    Bxa�,�  �          @�@(�@�=q@=qA�  B�L�@(�@�
=@�B&�\B��                                    Bxa�;�  �          @��@ff@Ӆ@��A�G�B���@ff@�Q�@�B&�
B��3                                    Bxa�JB  �          @�G�@,��@�\)?�p�A]�B��@,��@�p�@���B�By��                                    Bxa�X�  �          @�{@.�R@�{?���A8(�B�p�@.�R@���@u�B�HBv�\                                    Bxa�g�  "          @߮@Dz�@�Q�#�
����Bz�@Dz�@�Q�@p�A�G�Bw�R                                    Bxa�v4  "          @���@G�@ə���33�4z�B~�@G�@��
@�
A��Bx{                                    Bxa���  �          @�{@*=q@��;���
=B���@*=q@�p�@(�A���B��\                                    Bxa���  �          @޸R@*�H@�p���{�1�B��
@*�H@�
=@Q�A��
B��
                                    Bxa��&  T          @ڏ\@>�R@�(��Ǯ�Q�B�33@>�R@�\)@p�A�G�Bz33                                    Bxa���  �          @�33@"�\@ʏ\��
=�c�
B�Q�@"�\@�@G�A�
=B���                                    Bxa��r  �          @���@(Q�@��
    <��
B��@(Q�@���@*�HA���B�\                                    Bxa��  
�          @��@(Q�@ȣ׾\)��z�B�� @(Q�@�  @ ��A��B��
                                    Bxa�ܾ  
�          @�=q@��@�(��8Q�\B���@��@��
@!G�A�G�B��=                                    Bxa��d  
�          @ۅ?��H@�Q�?�Q�A ��B��\?��H@���@tz�B��B��H                                    Bxa��
  �          @�?���@���?E�@�z�B���?���@��R@`  A�{B��3                                    Bxa��  �          @�=q?�@�{?z�@��HB�.?�@�=q@UA���B�L�                                    Bxa�V  T          @ۅ?�(�@�
=?
=q@���B�z�?�(�@��
@U�A�(�B��\                                    Bxa�%�  T          @��H?��@�ff>�ff@qG�B���?��@�(�@O\)A�B���                                    Bxa�4�  �          @�33?�{@�{?(�@��
B���?�{@���@XQ�A�ffB��{                                    Bxa�CH  "          @�G�?��H@��R?\(�@��B��?��H@�G�@S33A��\B�u�                                    Bxa�Q�  �          @�=q?���@��H?(��@�\)B�\?���@��@J�HA�
=B���                                    Bxa�`�  �          @��?�{@�p�?!G�@��B��
?�{@�=q@L(�A�\B�p�                                    Bxa�o:  �          @��?��H@��?�@�ffB��?��H@��@O\)A�RB���                                    Bxa�}�  �          @�{?�ff@�ff>�z�@�B�?�ff@�p�@Mp�Aԏ\B��=                                    Bxa���  
�          @�z�?�@�z�=�G�?h��B�
=?�@�{@A�A���B���                                    Bxa��,  �          @ᙚ@G�@�Q�=u?�\B��@G�@��H@;�A�(�B�B�                                    Bxa���  �          @��@�
@�=q��(��\��B�ff@�
@˅@ ��A��HB��                                    Bxa��x  T          @�
=?��@��E����HB���?��@��H@  A���B�#�                                    Bxa��  
�          @�G�@   @�
=�E��\B�  @   @�(�@G�A��B�z�                                    Bxa���  T          @�G�@�@�R��R��
=B�.@�@���@ ��A�B�=q                                    Bxa��j  
�          @�(�@�@�녾Ǯ�?\)B���@�@љ�@*�HA�G�B�(�                                    Bxa��  
�          @�33@�R@�  �#�
�L��B���@�R@ʏ\@>�RA��B�33                                    Bxa��  T          @�Q�@�\@�
==�G�?Y��B�k�@�\@�\)@L(�A�
=B�                                    Bxa�\  �          @���?�(�@�(�>�?z�HB��?�(�@�z�@J=qAʏ\B��R                                    Bxa�  �          @�R@��@�33���ͿQ�B�ff@��@�\)@5A��B�#�                                    Bxa�-�  "          @�  @(�@���=#�
>�{B��q@(�@�
=@?\)A�p�B�#�                                    Bxa�<N  �          @�z�?��H@�>���@E�B�(�?��H@�  @Z=qAۮB�.                                    Bxa�J�  
�          @�\?��@�Q�?�z�AY�B��?��@�z�@���B�B��q                                    Bxa�Y�  	�          @ᙚ?��@�?��A5�B�\?��@�{@��
B�B���                                    Bxa�h@  
�          @�R?\)@�{@#33A�(�B�aH?\)@�ff@�B8ffB�                                    Bxa�v�  �          @��
?!G�@ָR@�\A�  B�W
?!G�@�=q@�ffB0Q�B��                                    Bxa���  T          @�33?0��@��@(Q�A�{B�{?0��@�G�@�ffB<=qB��R                                    Bxa��2  "          @�Q�?z�@�{@,(�A��\B���?z�@���@�ffB?�RB��3                                    Bxa���  
�          @޸R��@��@J�HA��B��{��@�{@���BR
=B�z�                                    Bxa��~  T          @أ�=L��@��@2�\A�
=B�
==L��@�33@�p�BG�B���                                    Bxa��$  T          @��?��H@�z�?�p�AI�B�\?��H@�(�@��HBffB�(�                                    Bxa���  "          @�  ?��@��?�z�A@��B�
=?��@�p�@�G�B=qB��f                                    Bxa��p  �          @��?�ff@ʏ\?���AJ�HB��q?�ff@��\@���B��B�=q                                    Bxa��  
�          @�\)?#�
@�?޸RA�=qB��3?#�
@��H@�(�B&�
B�.                                    Bxa���  �          @�33?��H@��H?�  A<��B��\?��H@�\)@i��B33B�=q                                    Bxa�	b  �          @��?�  @���?��AE��B�
=?�  @��@l(�B�B�\)                                    Bxa�  �          @�=q?}p�@���@��A���B�Ǯ?}p�@��@�=qB8
=B��                                    Bxa�&�  
�          @�
==���@��
@!G�A��
B�=���@aG�@��BO�B��q                                    Bxa�5T  �          @�>�  @��@#33Aՙ�B�B�>�  @]p�@��BQz�B���                                    Bxa�C�  
Z          @�Q�>.{@��@>{A�G�B�  >.{@HQ�@��Bcz�B��q                                    Bxa�R�  �          @��R<#�
@�\)@J�HBz�B���<#�
@8Q�@��Bnz�B�k�                                    Bxa�aF  �          @����#�
@�@Z�HB�B��;#�
@.�R@��Bwz�B��                                    Bxa�o�  "          @�녾�@��@c�
B��B�aH��@#�
@�{B~�
B�                                    Bxa�~�  �          @��H����@P  @�
=BW�B��׾���?z�H@�{B��B�k�                                    Bxa��8  "          @�G��8Q�@_\)@��RBG{B�W
�8Q�?��@��B���B�Ǯ                                    Bxa���  "          @����
=@P��@�z�BS�B�ff�
=?��\@��
B�L�B�Ǯ                                    Bxa���  "          @��ͿY��@'
=@�z�Bm��B���Y��>��
@��B��{C#�                                    Bxa��*  �          @��ͿaG�@#�
@�p�Bo�\B��aG�>�=q@�=qB�aHC#)                                    Bxa���  T          @�Q�\(�@8��@��Bc(�B�{�\(�?
=@��B�33C�                                     Bxa��v  �          @�(��Ǯ@Mp�@���BSffB��)�Ǯ?��
@�  B�ffBݸR                                    Bxa��  �          @�\)��
=@a�@g�B0��B��쿗
=?���@���B�\B��R                                    Bxa���  �          @����R@h��@\(�B'�BٸR���R?�\@�B�ffB�(�                                    Bxa�h  "          @�  ��=q@a�@s�
B/33B�Ǯ��=q?�  @�
=B�\C�=                                    Bxa�  
�          @���@G�@w
=B7�B��=�?�\)@��HB��Cٚ                                    Bxa��  
�          @�����H@&ff@��BHG�C�3��H>�@�ffB~�C(��                                    Bxa�.Z  "          @�{�1G�?�\@mp�BA��Cz��1G���@��B_�
C4�R                                    Bxa�=   �          @��?��\@z=q>���@��
B�\?��\@XQ�?�p�A��
B��                                    Bxa�K�  �          @�ff@�33@{���
��=qB��@�33@�G���33�@��B,p�                                    Bxa�ZL  T          @��\@��@Q��(��ř�B  @��@��׿@  ��z�B/��                                    Bxa�h�  �          @�{@�  @Fff���H���B(�@�  @i��������\)B)z�                                    Bxa�w�  �          @�G�@j=q@W
=�xQ��0z�B*G�@j=q@\(�?(�@�z�B-{                                    Bxa��>  T          @��@p��@P  ������B#z�@p��@G�?�G�A;
=B                                      Bxa���  
�          @Å@��@w
=>�ff@��B�@��@QG�@ffA��HBQ�                                    Bxa���  
�          @�p�@���@��
����ffB�H@���@���?���A;33BQ�                                    Bxa��0  
�          @�ff@�{@�G��   �|��B�H@�{@vff?�=qA*�HB�                                    Bxa���  "          @��@�(�@g��@  ��G�A�\@�(�@e�?k�@�(�A�ff                                    Bxa��|  �          @�{@�z�@dz῎{��A�33@�z�@l(�?z�@�33A�                                      Bxa��"  �          @�@�  @p  ������RB   @�  @y��?�@�=qB                                      Bxa���  �          @ᙚ@��H@s33����
=qB  @��H@w�?8Q�@���B(�                                    Bxa��n  �          @ٙ�@�Q�@{��������B
=@�Q�@qG�?��RA(z�B	�                                    Bxa�
  T          @���@�=q@r�\�@  ��=qB��@�=q@n�R?�  A(�B�                                    Bxa��  T          @��@�Q�@_\)�5��ffBz�@�Q�@\��?h��@�ffB {                                    Bxa�'`  �          @�
=@�(�@`�׿8Q���z�Bz�@�(�@^{?h��A ��B(�                                    Bxa�6  
�          @θR@�\)@0  ����  A���@�\)@:�H>�z�@#33A�(�                                    Bxa�D�  �          @�p�@�Q�@(Q쿅��Aď\@�Q�@4z�>k�@33A��H                                    Bxa�SR  �          @�33@�Q�@�H�xQ���\A�{@�Q�@&ff>L��?�z�A�ff                                    Bxa�a�  
�          @�
=@��R@'��B�\� ��A��H@��R@,(�>�@�G�A�                                    Bxa�p�  
�          @�z�@�@!G��5��z�A�\@�@%�>�@��A�R                                    Bxa�D  �          @�  @���@(�þ�Q��fffA�z�@���@!G�?Y��Az�A�=q                                    Bxa���  �          @�p�@��R@>�R>�\)@C�
B(�@��R@#33?���A�\)A��                                    Bxa���  �          @��@�p�@H��>�@�Q�B  @�p�@&ff?���A���A�                                    Bxa��6  �          @�Q�@��@�\��p��Z{A���@��@��L���{Aי�                                    Bxa���  
�          @��R@�z�?�ff��=q��G�A��@�z�@z��R����A���                                    Bxa�Ȃ  "          @�\)@���>������
@���@���?�p���Q��~�RAa                                    Bxa��(  �          @��?����R�\�#33�\)C��?��Ϳ��p  �]z�C�˅                                    Bxa���  �          @�{@G
=�
=q��\��RC��@G
=?���\���Aff                                    Bxa��t  T          @�\)?@  ���
�G���p�C���?@  �HQ����H�O\)C���                                    Bxa�  T          @��H>�
=��ff�{��p�C���>�
=�G
=�����W�RC��
                                    Bxa��  $          @���=#�
��33�����HC�=q=#�
�L���z=q�K{C�Y�                                    Bxa� f  
�          @��\>������ff��33C��)>���;��s33�Q
=C���                                    Bxa�/  �          @�
=?+����H�.{�{C���?+������Q��l��C���                                    Bxa�=�  
�          @�ff�.{�0���e�NC�q�.{�^�R��{33Ch�                                    Bxa�LX  
�          @�녾������x���uC�|)���;����©  CE�=                                    Bxa�Z�  "          @�33?����Q������
=C�9�?���(Q��mp��W�C�^�                                    Bxa�i�  �          @��>Ǯ���׿�����C���>Ǯ�O\)�Dz��-(�C�q�                                    Bxa�xJ  
(          @��\>��R��\)�����\C��>��R�b�\����C���                                    Bxa���  
�          @n{?�\�`  ������C��?�\�'��%��0�C�w
                                    Bxa���  �          @Tz�?xQ��<�Ϳ�  ��G�C�"�?xQ��(��p��*�
C���                                    Bxa��<  "          @��>�z���  ��G����C��>�z��AG��:�H�/��C��q                                    Bxa���  
(          @��>B�\�vff���H��{C�` >B�\�2�\�AG��<�C��                                    Bxa���  �          @��H?������ff�ظRC�*=?���,���l���T�
C�8R                                    Bxa��.  �          @���?O\)���\�{���C�
=?O\)�?\)��Q��W�C��)                                    Bxa���  �          @���?E���{��(����
C���?E��Tz��vff�B(�C���                                    Bxa��z  T          @��?�
=��������C�q�?�
=�G
=�l���@�C�b�                                    Bxa��   
Z          @�ff?s33��G���(���Q�C���?s33�R�\�c�
�8ffC��                                    Bxa�
�  
�          @��?n{���ÿ�������C���?n{�N{�k��>��C�f                                    Bxa�l  �          @���?L���������H��(�C���?L���L(��r�\�D�C�f                                    Bxa�(  T          @���?Y����녿����z�C�J=?Y���X���Z=q�1(�C��                                    Bxa�6�  
�          @��?fff���R��  ��C�?fff�E�N�R�4{C�#�                                    Bxa�E^  "          @�(�?���{��p���=qC�AH?��S33�w��D�C���                                    Bxa�T  "          @�G�?fff��녿�����p�C�Ff?fff�3�
�\���E��C��                                    Bxa�b�  "          @�ff?�p��qG���p�����C��?�p���R�]p��MG�C�:�                                    Bxa�qP  T          @�G�?����vff�	����ffC��=?�����R�i���U�C���                                    Bxa��  
�          @��\?#�
��=q��(��̏\C�s3?#�
�0���e�N�HC���                                    Bxa���  
�          @���?0�������   ��z�C���?0���:�H�l���KC��q                                    Bxa��B  
�          @�33?�\�������C�P�?�\�G
=�j�H�E\)C���                                    Bxa���  �          @�  >�������������\C���>����W��S�
�1=qC�Y�                                    Bxa���  
�          @���?�R�����R�w\)C�H?�R�Z=q�E�'
=C�/\                                    Bxa��4  
(          @��>\��p��h���9�C�u�>\�c33�333�Q�C��                                    Bxa���  
�          @��\�����mp��i���-�C}�H���ÿ�
=���Cmu�                                    Bxa��  �          @Y����
=��R�2�\�s=qCK�{��
=?0���1��p�
C�                                    Bxa��&  	�          @�33�����HQ��L���/=qCz=q���׿�\)��Q���Cf��                                    Bxa��  
�          @������n{�HQ��G�C|���녿����R�}�CoB�                                    Bxa�r  T          @�{��  �z=q�33���C�)��  �%��e��Q\)Cx��                                    Bxa�!  
�          @�Q�����=q�   ��=qC�k�����L(��u�H�C�O\                                    Bxa�/�  �          @�33�����R�0�����C�*=���!G���33�m{C���                                    Bxa�>d  T          @�z���~�R�0���
ffC��
���z���Q��s�HC�ff                                    Bxa�M
  T          @�33�\(�����"�\�홚C�n�\(��0����\)�^p�C|��                                    Bxa�[�  T          @�Q쿎{��
=�!G���
=C���{�7���Q��X�\Cx�\                                    Bxa�jV  T          @���J=q���R�%���\)C���J=q�5�����^�HC~ff                                    Bxa�x�  �          @��R�fff���
�1G��G�C���fff����=q�l
=Cy��                                    Bxa���  �          @�ff����y���������C~Ǯ����(Q��\(��I�HCx\                                    Bxa��H  �          @��R��=q�s33��
�܏\C~\��=q�{�c33�S{CvE                                    Bxa���  �          @�ff��Q������G���{C}�{��Q��.{�g��J�CvE                                    Bxa���  T          @����p��������ܸRCy���p��'��vff�O�HCpn                                    Bxa��:  
�          @�
=�޸R��녿�p���\)Cw�R�޸R�Dz��^�R�2G�Cpn                                    Bxa���  �          @�������{������Cw\����H���h���3��Co{                                    Bxa�߆  	�          @���������׿�
=�g�Cy�=�����R�\�>{�ffCt#�                                    Bxa��,  �          @�{����=q��
=��Q�C{�{���?\)�j�H�?�RCt�f                                    Bxa���  T          @����R���������C}޸���R�E�Z�H�7  Cx�                                    Bxa�x  
(          @�33�&ff���}p��G\)C��R�&ff�aG��8Q��=qC���                                    Bxa�  
�          @��
������\(��5p�C�o\���Y���*�H��C���                                    Bxa�(�  �          @���B�\������G�����C�� �B�\�J=q�Q��8{C�L�                                    Bxa�7j  
�          @�(�������(���\)���HC��
�����Dz��h���FC���                                    Bxa�F  �          @�(���{��{��{���C�\)��{�]p��b�\�0�C|J=                                    Bxa�T�  �          @��������=q�AG���HC~�Ϳ���� ����z��lz�Ct�                                     Bxa�c\  
�          @����=q���R�B�\�
(�C|uÿ�=q�����33�l��Cp��                                    Bxa�r  	�          @�(����r�\�N�R��\C|ٚ����Q����H�  Cn�)                                    Bxa���  �          @�녿���`  �_\)�.�C}������Ǯ��{� Clff                                    Bxa��N  "          @�33�s33�j�H�Z=q�'C�\�s33��  ��{\Cq�H                                    Bxa���  	�          @����fff�y���HQ���C��=�fff����G��}��Cv�q                                    Bxa���  �          @�(�����<(��o\)�H�Cy������s33�����C]�                                    Bxa��@  �          @��Ϳ�33�Tz��Y���0
=Cz�f��33��Q�������Cgp�                                    Bxa���  �          @�Q쿘Q��U��I���&��CzO\��Q�Ǯ����Ch�                                    Bxa�،  "          @�  ����W��>{��RCuaH�����
=����u(�Ccc�                                    Bxa��2  �          @��R�У��Y���3�
���CtT{�У׿��
��G��l\)Cck�                                    Bxa���  T          @������X���4z��{Cu+����ÿ�G���G��nCd\)                                    Bxa�~  T          @�{�c�
�n{�-p��
=C�@ �c�
�
=���H�r�HCw\                                    Bxa�$  "          @��H�c�
�w��0���=qC�� �c�
�{��ff�qp�Cx�                                    Bxa�!�  �          @��H��{�~{�>�R�(�C~W
��{�{��{�sQ�CsO\                                    Bxa�0p  �          @�33��p���ff�C33�p�C}����p��������o(�Cr��                                    Bxa�?  �          @�zῊ=q����H����C��=q�Q����R�s�Cu��                                    Bxa�M�  "          @���n{��(��<(��(�C����n{�%���H�l
=Cz0�                                    Bxa�\b  �          @�  �Tz���p��]p���C�b��Tz������R�qCy+�                                    Bxa�k  �          @�
=�Q���  �h���&�C�5ÿQ녿��������{Cw(�                                    Bxa�y�  �          @���c�
��{�Z�H�Q�C�׿c�
�p���{ffCx!H                                    Bxa��T  �          @�G��z�H���R�-p���p�C�  �z�H�?\)�����\��C{޸                                    Bxa���  �          @����������������C��R����Q���33�G��C|c�                                    Bxa���  �          @��׿0������Q����C��\�0���b�\�X���,��C�}q                                    Bxa��F  �          @�\)�����ÿ������C�:���<���Fff�7�HC��H                                    Bxa���  
�          @��k��a녿�����G�Cff�k��'��)���.Cz��                                    Bxa�ђ  T          @��R�aG��z=q�\��Q�C����aG��3�
�G
=�:33C|�\                                    Bxa��8  �          @�p�?E��8Q�@:�HB1�\C���?E��xQ�?���A��C���                                    Bxa���  �          @}p�?�  �.{@*�HB*��C��?�  �hQ�?�A�  C��
                                    Bxa���  "          @r�\?�R�(Q�@{B)�
C��=?�R�]p�?��
A��C��                                    Bxa�*  
�          @\)?�\)�z�@/\)B;\)C��H?�\)�S33?�A��HC�\)                                    Bxa��  T          @��H>��H�L��@\)B\)C�]q>��H�~�R?L��A3�C��f                                    Bxa�)v  
(          @��>���z�?��A��C�&f>���Q쾅��O\)C��f                                    Bxa�8  �          @�33?333��p�?��A�=qC���?333��  �\��  C�7
                                    Bxa�F�  "          @�Q�>�����p�?Tz�Ap�C���>������ÿ��H��\)C��3                                    Bxa�Uh  �          @��?B�\���\?��A��\C��
?B�\���ÿ+����C��                                    Bxa�d  �          @���?���?Tz�@XQ�B�z�B
�?����E�@Y��B�� C���                                    Bxa�r�  �          @��?�{���R@�\)B�\)C��=?�{��@w�B^G�C�s3                                    Bxa��Z  �          @�33?�p��}p�@�p�B�.C�
=?�p��7
=@`��B=�HC��H                                    